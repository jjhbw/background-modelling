#[macro_use]
extern crate criterion;
extern crate image;
extern crate image_manipulation;
extern crate rand;

use criterion::Criterion;
use image::Rgb;
use image::RgbImage;
use image_manipulation::bg_model::gmm::*;
use image_manipulation::bg_model::*;

// TODO: disable rayon (or force single-thread)
// TODO: model updating: check if we want models and their parameters to be re-used across iterations

/// RGB image to use in benchmarks. This is neither noise nor
/// similar to natural images - it's just a convenience method
/// to produce an image that's not constant.
/// Lifted from the imageproc crate.
pub fn rgb_bench_image(width: u32, height: u32, seed_a: u32, seed_b: u32) -> RgbImage {
    use std::cmp;
    let mut image = RgbImage::new(width, height);

    for y in 0..image.height() {
        for x in 0..image.width() {
            let r = (x % seed_a + y % seed_b) as u8;
            let g = 255u8 - r;
            let b = cmp::min(r, g);
            image.put_pixel(x, y, Rgb([r, g, b]));
        }
    }
    image
}

fn default_model_hyperparameters() -> GaussianMixtureSettings {
    return GaussianMixtureSettings {
        // alphamin can be considered 1 / T where T is the maximum size of the training set the model will remember
        alphamin: 0.0001,
        max_components: 5,
        initial_variance: 15.0,
        cf: 0.1,
        mahal_threshold: 3.0 * 3.0, // note: we provide D^2
    };
}

fn bench_model_update_single_image(c: &mut Criterion) {
    // build a test image
    let img = rgb_bench_image(200, 200, 1, 1);

    // initiate a new background subtraction model
    let settings = default_model_hyperparameters();
    let (width, height) = img.dimensions();
    let mut model = BackgroundModel::new(width, height, &settings);

    c.bench_function("model_update_single_image", move |b| {
        b.iter(|| model.update(&img))
    });
}

fn bench_model_update_single_pixel(c: &mut Criterion) {
    // build a test image
    let img = rgb_bench_image(1, 1, 1, 1);

    // initiate a new background subtraction model
    let settings = default_model_hyperparameters();
    let (width, height) = img.dimensions();
    let mut model = BackgroundModel::new(width, height, &settings);

    c.bench_function("model_update_single_pixel", move |b| {
        b.iter(|| model.update(&img))
    });
}

fn bench_model_update_vs_image_size(c: &mut Criterion) {
    // initiate a new background subtraction model
    let settings = default_model_hyperparameters();

    c.bench_function_over_inputs(
        "model_update_vs_image_size",
        move |b, dimensions| {
            let width: u32 = dimensions.0;
            let height: u32 = dimensions.1;
            let img = rgb_bench_image(width, height, 1, 1);
            let mut model = BackgroundModel::new(width, height, &settings);
            b.iter(|| model.update(&img))
        },
        vec![(320, 240), (640, 480), (1280, 720)],
    );
}

fn bench_model_predict_single_image(c: &mut Criterion) {
    // initiate a new background subtraction model
    let settings = default_model_hyperparameters();
    let width = 200;
    let height = 200;
    let mut model = BackgroundModel::new(width, height, &settings);

    // train on pseudorandom test images
    for i in 1..100 {
        model.update(&rgb_bench_image(width, height, i, i + 1));
    }

    // generate the image to predict
    let testimage = rgb_bench_image(width, height, 40, 60);

    c.bench_function("model_predict_single_image", move |b| {
        b.iter(|| model.predict(&testimage))
    });
}

fn bench_model_predict_single_pixel(c: &mut Criterion) {
    // initiate a new background subtraction model
    let settings = default_model_hyperparameters();
    let width = 1;
    let height = 1;
    let mut model = BackgroundModel::new(width, height, &settings);

    // train on pseudorandom test images
    for i in 1..100 {
        model.update(&rgb_bench_image(width, height, i, i + 1));
    }

    // generate the image to predict
    let testimage = rgb_bench_image(width, height, 40, 60);

    c.bench_function("model_predict_single_pixel", move |b| {
        b.iter(|| model.predict(&testimage))
    });
}

fn bench_model_predict_vs_image_size(c: &mut Criterion) {
    let settings = default_model_hyperparameters();

    c.bench_function_over_inputs(
        "model_predict_vs_image_size",
        move |b, dimensions| {
            // initiate a new background subtraction model
            let width: u32 = dimensions.0;
            let height: u32 = dimensions.1;
            let mut model = BackgroundModel::new(width, height, &settings);

            // train on pseudorandom test images
            for i in 1..100 {
                model.update(&rgb_bench_image(width, height, i, i + 1));
            }

            // generate the image to predict
            let testimage = rgb_bench_image(width, height, 40, 60);

            b.iter(|| model.predict(&testimage))
        },
        // TODO: currently larger images lead to extremely long benchmark times
        // probably due to Criterion-related bug.
        vec![(320, 240), (640, 480), (1280, 720)],
    );
}

// // Utility functions
fn bench_pixel_to_vector(c: &mut Criterion) {
    // build a test pixel
    let pix = Rgb([100, 100, 100]);

    c.bench_function("pixel_to_vector", move |b| b.iter(|| pixel_to_vector(&pix)));
}



criterion_group!(utils, bench_pixel_to_vector);
criterion_group!(
    model_updating,
    bench_model_update_single_image,
    bench_model_update_single_pixel,
    bench_model_update_vs_image_size,
    bench_model_predict_single_image,
    bench_model_predict_single_pixel,
    bench_model_predict_vs_image_size
);
criterion_main!(utils, model_updating);
