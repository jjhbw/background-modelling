extern crate glob;
extern crate image;
extern crate nalgebra;
extern crate rayon;

use glob::glob;
use image::DynamicImage;
use image::DynamicImage::ImageRgb8;
use image::RgbImage;
use std::path::Path;
use std::path::PathBuf;

pub mod bg_model;

pub fn get_test_data(n: usize) -> Vec<PathBuf> {
    let allfiles: Vec<PathBuf> = glob("./example_data/*.jpg")
        .expect("could not read files")
        
        // remove all unreadable paths
        .filter_map(Result::ok)

        .collect();

    // take the first n
    let testfiles = allfiles[0..n].to_vec();

    if testfiles.len() != n {
        panic!("could not get the desired number of test files")
    }

    return testfiles;
}

// takes a vector of generics: 'everything that looks and quacks like a std::path::Path'
pub fn import_images<P: AsRef<Path>>(paths: &Vec<P>) -> Vec<RgbImage> {
    // load images into memory from file paths
    let mut images: Vec<RgbImage> = Vec::new();

    for path in paths {
        let img = image::open(path).unwrap();

        // Write the contents of this image to the Writer in PNG format.
        images.push(to_rgb(img))
    }

    println!("slurped {} images into memory", images.len());

    // check that all images have the same dimensions
    let dims: Vec<(u32, u32)> = images.iter().map(|img| return img.dimensions()).collect();
    // println!("Image dimensions: {:?}", dims);
    let first_img_dims = dims[0];
    for dimen in dims {
        if dimen != first_img_dims {
            panic!("image dimensions not equal")
        }
    }

    return images;
}

fn to_rgb(img: DynamicImage) -> RgbImage {
    match img {
        ImageRgb8(x) => x,
        _ => panic!("only 3-channel RGB images have been implemented"),
    }
}
