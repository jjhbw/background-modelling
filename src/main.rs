use std::env;

extern crate image;
extern crate imageproc;
extern crate image_manipulation;

// use image::RgbImage;
// use image::Rgb;
// use imageproc::morphology::close;
// use imageproc::distance_transform::Norm;
use image_manipulation::bg_model::gmm::*;
use image_manipulation::bg_model::*;

fn main() {
    // Collect all elements in the iterator that contains the command line arguments
    let args: Vec<String> = env::args().collect();

    // remove the first element from the list of arguments, which is the call to the binary
    let inputfiles = &args[1..];
    let images = image_manipulation::import_images(&inputfiles.to_vec());

    let settings = GaussianMixtureSettings {
        // alphamin can be considered 1 / T where T is the maximum size of the training set the model will remember
        alphamin: 0.0001,
        max_components: 5,
        initial_variance: 15.0,
        cf: 0.3,
        mahal_threshold :  3.0 * 3.0, // note: we provide D^2
    };

    let (width, height) = images[0].dimensions();
    let mut model = BackgroundModel::new(width, height, &settings);

    for (i, img) in images.iter().enumerate() {
        // update the model with the image
        model.update(&img);

        // add leading zeroes for easier downstream proc with ffmpeg
        let identifier = format!("{:<04}", i);

        // predict the background of the new image and store the background prediction map
        model.predict(img)
            .save(format!("testimg_predicted_{}.png", identifier)).unwrap();
        
        // store the background model
        // model.get_background_estimate()
        //     .save(format!("background_estimate_{}.png", identifier)).unwrap();

        // image_subtract(&img, &background_estimate)
            // .save(format!("subtracted_{}.png", i)).unwrap();

        // apply morphological closing
        // close(&predicted, Norm::LInf, 1).save("test_image_predicted_postprocessed.png").unwrap();

        println!("Processed sample image no. {}", i);
    }
   
}

// subtracts b from a
// fn image_subtract(a: &RgbImage, b: &RgbImage) -> RgbImage{
//     assert_eq!(a.width(), b.width());
//     assert_eq!(a.height(), b.height());
//     let mut new = RgbImage::new(a.width() as u32, a.height() as u32);
//     for (x, y, pixel) in new.enumerate_pixels_mut(){
        
//         *pixel = Rgb{
//             data: [
//                 a.get_pixel(x,y)[0].saturating_sub(b.get_pixel(x,y)[0]),
//                 a.get_pixel(x,y)[1].saturating_sub(b.get_pixel(x,y)[1]),
//                 a.get_pixel(x,y)[2].saturating_sub(b.get_pixel(x,y)[2])
//             ]
//         }
//     }

//     return new
// }