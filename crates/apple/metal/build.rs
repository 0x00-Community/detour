fn main() {
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR")).join("donor.bin");
    println!("cargo::rerun-if-changed=src/donor.metal");
    println!("cargo::rerun-if-changed=build.rs");
    let bytes = if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") { archive::make().unwrap_or_else(|e| {
        println!("cargo::warning=no donor archive: {e}");
        Vec::new()
    }) } else { Vec::new() };
    std::fs::write(&out, bytes).expect("write donor.bin");
}

#[cfg(target_os = "macos")]
mod archive {
    use objc2_foundation::{NSString, NSURL};
    use objc2_metal::{MTLBinaryArchive, MTLBinaryArchiveDescriptor, MTLComputePipelineDescriptor, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary};

    pub fn make() -> Result<Vec<u8>, String> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        let arch = device.architecture().name().to_string();
        if !arch.starts_with("applegpu_g16") {
            return Err(format!("{arch}: not an Apple9 GPU"));
        }
        let src = std::fs::read_to_string("src/donor.metal").map_err(|e| e.to_string())?;
        let library = device.newLibraryWithSource_options_error(&NSString::from_str(&src), None).map_err(|e| e.to_string())?;
        let donor = library.newFunctionWithName(&NSString::from_str("donor")).ok_or("no donor")?;
        let ar = device.newBinaryArchiveWithDescriptor_error(&MTLBinaryArchiveDescriptor::new()).map_err(|e| e.to_string())?;
        let cd = MTLComputePipelineDescriptor::new();
        cd.setComputeFunction(Some(&donor));
        ar.addComputePipelineFunctionsWithDescriptor_error(&cd).map_err(|e| e.to_string())?;
        let path = std::path::PathBuf::from(std::env::var("OUT_DIR").map_err(|e| e.to_string())?).join("donor-serialized.bin");
        ar.serializeToURL_error(&NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()))).map_err(|e| e.to_string())?;
        std::fs::read(&path).map_err(|e| e.to_string())
    }
}

#[cfg(not(target_os = "macos"))]
mod archive {
    pub fn make() -> Result<Vec<u8>, String> {
        Err("not macOS".into())
    }
}
