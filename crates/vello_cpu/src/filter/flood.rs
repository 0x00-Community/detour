use super::FilterEffect;
use crate::vello_cpu::filter::context::ScratchBuffer;
use vello_common::filter::flood::Flood;
use vello_common::pixmap::Pixmap;

impl FilterEffect for Flood {
    fn execute_lowp(&self, pixmap: &mut Pixmap, _filter_scratch: &mut ScratchBuffer) {
        pixmap.data_mut().fill(self.color.premultiply().to_rgba8());
    }

    fn execute_highp(&self, pixmap: &mut Pixmap, _filter_scratch: &mut ScratchBuffer) {
        pixmap.data_mut().fill(self.color.premultiply().to_rgba8());
    }
}

#[cfg(any())]
mod tests {
    use super::*;
    use crate::vello_cpu::color::AlphaColor;
    use crate::vello_cpu::filter::context::ScratchBuffer;
    use vello_common::color::{PremulRgba8, Srgb};

    #[test]
    fn test_flood_semi_transparent_lowp() {
        let mut pixmap = Pixmap::new(2, 2);
        let mut filter_scratch = ScratchBuffer::new();

        let color = AlphaColor {
            components: [1.0, 1.0, 1.0, 0.5],
            cs: std::marker::PhantomData::<Srgb>,
        };
        let flood = Flood::new(color);
        flood.execute_lowp(&mut pixmap, &mut filter_scratch);

        for y in 0..2 {
            for x in 0..2 {
                let pixel = pixmap.sample(x, y);
                assert_eq!(
                    pixel,
                    PremulRgba8 {
                        r: 128,
                        g: 128,
                        b: 128,
                        a: 128
                    }
                );
            }
        }
    }

    #[test]
    fn test_flood_semi_transparent_highp() {
        let mut pixmap = Pixmap::new(2, 2);
        let mut filter_scratch = ScratchBuffer::new();

        let color = AlphaColor {
            components: [1.0, 1.0, 1.0, 0.5],
            cs: std::marker::PhantomData::<Srgb>,
        };
        let flood = Flood::new(color);
        flood.execute_highp(&mut pixmap, &mut filter_scratch);

        for y in 0..2 {
            for x in 0..2 {
                let pixel = pixmap.sample(x, y);
                assert_eq!(
                    pixel,
                    PremulRgba8 {
                        r: 128,
                        g: 128,
                        b: 128,
                        a: 128
                    }
                );
            }
        }
    }
}
