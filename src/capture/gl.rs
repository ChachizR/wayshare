//! Surfaceless EGL + desktop GL 4.3 context for compute passes over dmabufs.

use std::ffi::c_void;
use std::os::fd::RawFd;

use anyhow::{Context, Result, anyhow, bail};
use glow::HasContext;
use khronos_egl as egl;

const EGL_PLATFORM_SURFACELESS_MESA: egl::Enum = 0x31dd;
const EGL_LINUX_DMA_BUF_EXT: egl::Enum = 0x3270;
const EGL_LINUX_DRM_FOURCC_EXT: egl::Attrib = 0x3271;
const EGL_CONTEXT_OPENGL_PROFILE_MASK: egl::Int = 0x30fd;
const EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT: egl::Int = 1;

type ImageTargetTexture2D = unsafe extern "system" fn(target: u32, image: *mut c_void);

/// One plane of a dmabuf, as EGL wants it.
pub struct PlaneDesc {
    pub fd: RawFd,
    pub fourcc: u32,
    pub width: u32,
    pub height: u32,
    pub offset: u32,
    pub pitch: u32,
    pub modifier: u64,
}

/// EGL plane attribute keys are contiguous for planes 0-2, but plane 3 was added
/// by EGL_EXT_image_dma_buf_import_modifiers in a separate range.
const PLANE_ATTRIBS: [[egl::Attrib; 5]; 4] = [
    [0x3272, 0x3273, 0x3274, 0x3443, 0x3444],
    [0x3275, 0x3276, 0x3277, 0x3445, 0x3446],
    [0x3278, 0x3279, 0x327a, 0x3447, 0x3448],
    [0x3440, 0x3441, 0x3442, 0x3449, 0x344a],
];

/// Import all memory planes, including auxiliary compression planes. INVALID
/// means implicit layout, so no plane receives explicit modifier attributes.
fn dmabuf_attribs(planes: &[PlaneDesc]) -> Result<Vec<egl::Attrib>> {
    if planes.is_empty() || planes.len() > PLANE_ATTRIBS.len() {
        bail!(
            "EGL dmabuf import requires 1-4 planes; got {}",
            planes.len()
        );
    }
    let p = &planes[0];
    let mut attribs = vec![
        egl::WIDTH as egl::Attrib,
        p.width as egl::Attrib,
        egl::HEIGHT as egl::Attrib,
        p.height as egl::Attrib,
        EGL_LINUX_DRM_FOURCC_EXT,
        p.fourcc as egl::Attrib,
    ];
    for (p, keys) in planes.iter().zip(PLANE_ATTRIBS) {
        attribs.extend([
            keys[0],
            p.fd as egl::Attrib,
            keys[1],
            p.offset as egl::Attrib,
            keys[2],
            p.pitch as egl::Attrib,
        ]);
        if p.modifier != super::alloc::MOD_INVALID {
            attribs.extend([
                keys[3],
                (p.modifier & 0xffff_ffff) as egl::Attrib,
                keys[4],
                (p.modifier >> 32) as egl::Attrib,
            ]);
        }
    }
    attribs.push(egl::ATTRIB_NONE);
    Ok(attribs)
}

pub struct Gl {
    egl: egl::Instance<egl::Static>,
    dpy: egl::Display,
    ctx: egl::Context,
    pub gl: glow::Context,
    image_target: ImageTargetTexture2D,
}

impl Gl {
    pub fn new() -> Result<Self> {
        let egl = egl::Instance::new(egl::Static);
        // SAFETY: `DEFAULT_DISPLAY` is the null native display, which the surfaceless platform
        // expects, and the attribute list is terminated by `ATTRIB_NONE`.
        let dpy = unsafe {
            egl.get_platform_display(
                EGL_PLATFORM_SURFACELESS_MESA,
                egl::DEFAULT_DISPLAY,
                &[egl::ATTRIB_NONE],
            )
        }
        .context("eglGetPlatformDisplay(surfaceless)")?;
        egl.initialize(dpy)?;
        egl.bind_api(egl::OPENGL_API)?;
        // SAFETY: A null config handle is only passed through to eglCreateContext (no config is
        // requested); it is never dereferenced here.
        let no_config = unsafe { egl::Config::from_ptr(std::ptr::null_mut()) };
        let ctx = egl.create_context(
            dpy,
            no_config,
            None,
            &[
                egl::CONTEXT_MAJOR_VERSION,
                4,
                egl::CONTEXT_MINOR_VERSION,
                3,
                EGL_CONTEXT_OPENGL_PROFILE_MASK,
                EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
                egl::NONE,
            ],
        )?;
        egl.make_current(dpy, None, None, Some(ctx))?;
        // SAFETY: The context created above is current on this thread, and the loader returns
        // pointers from eglGetProcAddress (or null for unknown names).
        let gl = unsafe {
            glow::Context::from_loader_function(|s| {
                egl.get_proc_address(s)
                    .map_or(std::ptr::null(), |f| f as *const c_void)
            })
        };
        let image_target = egl
            .get_proc_address("glEGLImageTargetTexture2DOES")
            .ok_or_else(|| anyhow!("no glEGLImageTargetTexture2DOES"))?;
        // SAFETY: `image_target` is the non-null pointer eglGetProcAddress returned for
        // glEGLImageTargetTexture2DOES, whose C signature is `void (GLenum, GLeglImageOES)`,
        // matching `ImageTargetTexture2D`.
        let image_target: ImageTargetTexture2D = unsafe { std::mem::transmute(image_target) };
        let gl = Self {
            egl,
            dpy,
            ctx,
            gl,
            image_target,
        };
        gl.release()?;
        Ok(gl)
    }

    /// Made current per use so the owner may move between threads.
    pub fn make_current(&self) -> Result<()> {
        Ok(self
            .egl
            .make_current(self.dpy, None, None, Some(self.ctx))?)
    }

    pub fn release(&self) -> Result<()> {
        Ok(self.egl.make_current(self.dpy, None, None, None)?)
    }

    pub fn program(&self, src: &str) -> Result<glow::Program> {
        let gl = &self.gl;
        // SAFETY: Callers make this context current on the calling thread first, and every
        // handle used here was created from `self.gl`.
        unsafe {
            let shader = gl
                .create_shader(glow::COMPUTE_SHADER)
                .map_err(|e| anyhow!(e))?;
            gl.shader_source(shader, src);
            gl.compile_shader(shader);
            if !gl.get_shader_compile_status(shader) {
                bail!("compute shader: {}", gl.get_shader_info_log(shader));
            }
            let program = gl.create_program().map_err(|e| anyhow!(e))?;
            gl.attach_shader(program, shader);
            gl.link_program(program);
            gl.delete_shader(shader);
            if !gl.get_program_link_status(program) {
                bail!("compute program: {}", gl.get_program_info_log(program));
            }
            Ok(program)
        }
    }

    /// Wraps one dmabuf plane as a 2D texture (current context required).
    pub fn import(&self, p: &PlaneDesc) -> Result<(egl::Image, glow::Texture)> {
        self.import_planes(std::slice::from_ref(p))
    }

    /// Wraps a complete dmabuf, including compression planes, as a texture.
    /// The context must be current and all plane fds must remain open during import.
    pub fn import_planes(&self, planes: &[PlaneDesc]) -> Result<(egl::Image, glow::Texture)> {
        let attribs = dmabuf_attribs(planes)?;
        let image = self
            .egl
            .create_image(
                self.dpy,
                // SAFETY: EGL_LINUX_DMA_BUF_EXT images take EGL_NO_CONTEXT; the handle is only
                // passed through, never dereferenced.
                unsafe { egl::Context::from_ptr(egl::NO_CONTEXT) },
                EGL_LINUX_DMA_BUF_EXT,
                // SAFETY: EGL_LINUX_DMA_BUF_EXT images take a null client buffer; the handle is
                // only passed through, never dereferenced.
                unsafe { egl::ClientBuffer::from_ptr(std::ptr::null_mut()) },
                &attribs,
            )
            .context("eglCreateImage(dmabuf)")?;
        let gl = &self.gl;
        // SAFETY: The context is current on this thread (callers of `import` make it current),
        // and `image` is a live EGLImage created on `self.dpy` just above, so passing it to the
        // resolved glEGLImageTargetTexture2DOES is valid.
        let tex = unsafe {
            let t = gl.create_texture().map_err(|e| anyhow!(e))?;
            gl.bind_texture(glow::TEXTURE_2D, Some(t));
            (self.image_target)(glow::TEXTURE_2D, image.as_ptr());
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::NEAREST as i32,
            );
            t
        };
        Ok((image, tex))
    }

    /// Frees an imported texture (current context required).
    pub fn forget(&self, (image, tex): (egl::Image, glow::Texture)) {
        // SAFETY: The caller has this context current (see the doc comment), and `tex` was
        // created by `import` on this context and is deleted only here.
        unsafe { self.gl.delete_texture(tex) };
        let _ = self.egl.destroy_image(self.dpy, image);
    }
}

impl Drop for Gl {
    fn drop(&mut self) {
        let _ = self.egl.make_current(self.dpy, None, None, None);
        let _ = self.egl.destroy_context(self.dpy, self.ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::alloc::{MOD_INVALID, MOD_LINEAR};

    fn plane(modifier: u64) -> PlaneDesc {
        PlaneDesc {
            fd: 3,
            fourcc: 0x3432_5258,
            width: 64,
            height: 32,
            offset: 0,
            pitch: 256,
            modifier,
        }
    }

    fn has_modifier(attribs: &[egl::Attrib]) -> bool {
        attribs
            .iter()
            .step_by(2)
            .any(|&a| a == PLANE_ATTRIBS[0][3] || a == PLANE_ATTRIBS[0][4])
    }

    #[test]
    fn compressed_rgb_import_includes_all_memory_planes() {
        let modifier = 0x0100_0000_0000_0008;
        let mut planes = [plane(modifier), plane(modifier), plane(modifier)];
        for (i, p) in planes.iter_mut().enumerate() {
            p.fd = 10 + i as i32;
            p.offset = [0, 14745600, 14807040][i];
            p.pitch = [10240, 1280, 64][i];
        }
        let attrs = dmabuf_attribs(&planes).unwrap();
        let value = |key| {
            attrs
                .as_chunks::<2>()
                .0
                .iter()
                .find(|p| p[0] == key)
                .unwrap()[1]
        };
        for (p, keys) in planes.iter().zip(PLANE_ATTRIBS) {
            assert_eq!(value(keys[0]), p.fd as egl::Attrib);
            assert_eq!(value(keys[1]), p.offset as egl::Attrib);
            assert_eq!(value(keys[2]), p.pitch as egl::Attrib);
            assert_eq!(value(keys[3]), 8);
            assert_eq!(value(keys[4]), 0x0100_0000);
        }
        assert_eq!(attrs.last(), Some(&egl::ATTRIB_NONE));
    }

    #[test]
    fn fourth_plane_uses_extension_attribute_range() {
        let planes = [
            plane(MOD_LINEAR),
            plane(MOD_LINEAR),
            plane(MOD_LINEAR),
            plane(MOD_LINEAR),
        ];
        let attrs = dmabuf_attribs(&planes).unwrap();
        for key in [0x3440, 0x3441, 0x3442, 0x3449, 0x344a] {
            assert!(attrs.as_chunks::<2>().0.iter().any(|pair| pair[0] == key));
        }
        let implicit = [plane(MOD_INVALID), plane(MOD_INVALID)];
        let attrs = dmabuf_attribs(&implicit).unwrap();
        for keys in PLANE_ATTRIBS {
            assert!(
                !attrs
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .any(|p| p[0] == keys[3] || p[0] == keys[4])
            );
        }
    }

    #[test]
    fn invalid_plane_counts_are_rejected() {
        assert!(dmabuf_attribs(&[]).is_err());
        let planes: Vec<_> = (0..5).map(|_| plane(MOD_LINEAR)).collect();
        assert!(dmabuf_attribs(&planes).is_err());
    }

    #[test]
    fn invalid_modifier_imports_implicitly() {
        let attribs = dmabuf_attribs(&[plane(MOD_INVALID)]).unwrap();
        assert!(!has_modifier(&attribs));
        assert_eq!(attribs.last(), Some(&egl::ATTRIB_NONE));
    }

    #[test]
    fn explicit_modifier_is_passed_through() {
        let tiled = 0x0200_0000_1234_5678;
        for modifier in [MOD_LINEAR, tiled] {
            let attribs = dmabuf_attribs(&[plane(modifier)]).unwrap();
            let lo = attribs
                .iter()
                .position(|&a| a == PLANE_ATTRIBS[0][3])
                .unwrap();
            let hi = attribs
                .iter()
                .position(|&a| a == PLANE_ATTRIBS[0][4])
                .unwrap();
            assert_eq!(attribs[lo + 1], (modifier & 0xffff_ffff) as egl::Attrib);
            assert_eq!(attribs[hi + 1], (modifier >> 32) as egl::Attrib);
        }
    }
}
