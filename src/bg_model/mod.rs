use bg_model::gmm::GaussianMixtureModel;
use bg_model::gmm::GaussianMixtureSettings;
use image::GrayImage;
use image::Luma;
use image::Rgb;
use image::RgbImage;
use nalgebra::Vector3;
use rayon::prelude::*;
use std::sync::mpsc::channel;

pub mod gmm;

// TODO: currently, conversion between Pixels and Vector3<f32> is done twice: during update and prediction. Maybe consolidate at a higher level?
// TODO: apply and propagate generic image size constraints

// (i.e. dims image fed to train == dims of image fed to predict)
pub struct BackgroundModel {
    // a GMM for each pixel, stored as a 2-D vector X<Y<gmm>>
    pixel_models: Vec<(u32,u32,GaussianMixtureModel)>,
    width: u32,
    height: u32,
}

// using the RGB struct instead of a pixel abstraction allows us to rely on the 3-sized u8 array backing it.
pub fn pixel_to_vector(pixel: &Rgb<u8>) -> Vector3<f32> {
    return Vector3::new(
        pixel.data[0] as f32,
        pixel.data[1] as f32,
        pixel.data[2] as f32,
    );
}

pub fn vector_to_pixel(vec: &Vector3<f32>) -> Rgb<u8> {
    return Rgb {
        data: [vec[0] as u8, vec[1] as u8, vec[2] as u8],
    };
}

impl BackgroundModel {
    pub fn new(width: u32, height: u32, gmm_settings: &GaussianMixtureSettings) -> BackgroundModel {
        let mut model = BackgroundModel {
            pixel_models: Vec::new(),
            width: width,
            height: height,
        };

        for x in 0..width {
            for y in 0..height {
                model.pixel_models.push((x,y,GaussianMixtureModel::new(&gmm_settings)))
            }
        }

        return model;
    }

    pub fn update(&mut self, frame: &RgbImage) {
        self.pixel_models
            .par_iter_mut()
            .for_each(|(x, y, model)| {
                model.update(&pixel_to_vector(frame.get_pixel(*x, *y)));
            });
    }

    // returns a probability map as a grayscale image.
    pub fn predict(&self, frame: &RgbImage) -> GrayImage {

        // initiate a channel to send predicted pixels over
        // NOTE: instead of using a channel to explicitly synchronize the processing of each
        // individual pixel's prediction, we could also exert some more effort to 
        // convince the compiler that the desired operation does not produce data races.
        // TODO: we are not using the same Rayon pattern as with the .update() method, 
        // because the ImageBuffer struct does not allow iterating like that.
        let (sender, receiver) = channel();

        self.pixel_models
            .par_iter()
            .for_each_with(sender, |s, (x, y, model)| {
                let pixel = frame.get_pixel(*x,*y);
                
                // compute the background and foreground probabilities for the pixel
                let (bg, fg) = model.probabilities(&pixel_to_vector(&pixel));

                // normalize the probability to be expressed in u8
                let bg_over_fg = bg / fg;
                let background: u8 = if bg_over_fg > 1.0 { 0 } else { 1 };

                #[cfg(debug_assertions)]
                {
                    println!(
                        "x:{}, y:{}, pixel: {:?}, bg_probability: {}, fg_probability: {}, bg_over_fg:{}, background mask decision: {}, || {}",
                        x,
                        y,
                        pixel.data.to_vec(),
                        bg,
                        fg,
                        bg_over_fg,
                        bg_over_fg as u8,
                        model.summary()
                    );
                }

                debug_assert!(
                    bg <= 1.0 && bg >= 0.0,
                    "pixel background probability not between 0.0 and 1.0"
                );

                // return the predicted value and coordinates
                s.send((*x, *y, Luma([background * 255 as u8]))).unwrap();
                
            });

        // iterate over the receiver of the channel to build a probability map out of the predictions
        let mut probability_map = GrayImage::new(frame.width(), frame.height());
        for (x, y, prediction) in receiver {
            probability_map.put_pixel(x,y,prediction)
        }

        return probability_map;
    }


    pub fn get_background_estimate(&self) -> RgbImage {
        let width = self.width;
        let height = self.height;
        let mut background = RgbImage::new(width as u32, height as u32);

        for (x, y, model) in self.pixel_models.iter() {
            let mu = model.get_heaviest_mean();
            background.put_pixel(*x,*y, vector_to_pixel(mu));
        }

        return background;
    }
}

#[cfg(test)]
mod tests {

    // use all symbols found in the rest of this file.
    use super::*;

    #[test]
    fn test_pixel_to_fixed_length_vector() {
        let mut testimg = RgbImage::new(60, 120);
        let r = 1;
        let g = 2;
        let b = 3;

        for pixel in testimg.pixels_mut() {
            *pixel = Rgb([r, g, b]);
        }

        let testpixel = testimg.get_pixel(1, 1);
        println!("{}", pixel_to_vector(testpixel));

        assert_eq!(
            Vector3::new(r as f32, g as f32, b as f32),
            pixel_to_vector(testpixel)
        );
    }

}
