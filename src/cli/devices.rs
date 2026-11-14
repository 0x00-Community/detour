use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use clap::Args;

#[derive(Args)]
pub struct Devices {
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Driver {
    Xe,
    I915,
    Amdgpu,
    Radeon,
    Asahi,
    Nouveau,
    Nvidia,
    Virtio,
    Other,
}

const DRIVERS: [(Driver, &str); 8] = [
    (Driver::Xe, "xe"),
    (Driver::I915, "i915"),
    (Driver::Amdgpu, "amdgpu"),
    (Driver::Radeon, "radeon"),
    (Driver::Asahi, "asahi"),
    (Driver::Nouveau, "nouveau"),
    (Driver::Nvidia, "nvidia"),
    (Driver::Virtio, "virtio_gpu"),
];

impl From<&str> for Driver {
    fn from(name: &str) -> Driver {
        DRIVERS.iter().find(|(_, n)| *n == name).map_or(Driver::Other, |&(d, _)| d)
    }
}

enum Support {
    Native { model: String, state: &'static str },
    #[cfg_attr(all(feature = "xe3", feature = "agx-g13", feature = "rdna2"), expect(dead_code, reason = "every model built in"))]
    NotBuilt { feature: &'static str },
    Unsupported(String),
}

struct Node {
    path: PathBuf,
    driver: Driver,
    driver_name: String,
    pci: Option<(u16, u16)>,
    compatible: Vec<String>,
}

impl Node {
    fn read(path: PathBuf) -> Node {
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let device = Path::new("/sys/class/drm").join(name).join("device");
        let driver_name = fs::read_link(device.join("driver"))
            .ok()
            .and_then(|l| l.file_name().map(|n| n.to_string_lossy().into_owned()))
            .unwrap_or_default();
        let hex = |file: &str| {
            fs::read_to_string(device.join(file)).ok().and_then(|s| u16::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
        };
        let compatible = fs::read(device.join("of_node/compatible"))
            .map(|b| b.split(|&c| c == 0).filter(|s| !s.is_empty()).map(|s| String::from_utf8_lossy(s).into_owned()).collect())
            .unwrap_or_default();
        Node {
            driver: Driver::from(driver_name.as_str()),
            pci: hex("vendor").zip(hex("device")),
            path,
            driver_name,
            compatible,
        }
    }

    fn support(&self, verbose: bool) -> Support {
        match self.driver {
            Driver::Xe => self.xe(verbose),
            Driver::Amdgpu => self.amdgpu(verbose),
            Driver::Asahi => self.asahi(verbose),
            Driver::I915 => Support::Unsupported("i915: detour drives Intel through xe only (Xe3)".into()),
            _ => Support::Unsupported(format!("no detour driver for {}", self.driver_name)),
        }
    }

    fn xe(&self, verbose: bool) -> Support {
        if self.pci.is_some_and(|(_, d)| d != 0xb0a0) {
            return Support::Unsupported("xe, but detour targets Xe3 (8086:b0a0) only".into());
        }
        #[cfg(feature = "xe3")]
        {
            use detour::base::Gpu;
            match detour::xe::Xe::open(&self.path) {
                Ok(xe) => {
                    if verbose {
                        println!("{:#?}", xe.driver_info());
                    }
                    Support::Native { model: "Intel Xe3".into(), state: "verified on the GPU" }
                }
                Err(e) => Support::Unsupported(format!("opening it failed: {e}")),
            }
        }
        #[cfg(not(feature = "xe3"))]
        {
            let _ = verbose;
            Support::NotBuilt { feature: "xe3" }
        }
    }

    fn amdgpu(&self, verbose: bool) -> Support {
        #[cfg(feature = "rdna2")]
        {
            match detour::amdgpu::Amdgpu::open(&self.path) {
                Ok(amd) => {
                    use detour::base::Gpu;
                    let info = amd.driver_info();
                    if verbose {
                        println!("{info:#?}");
                    }
                    match info.gfx.version {
                        (10, 3) => Support::Native {
                            model: "AMD RDNA2 (GFX 10.3)".into(),
                            state: "compute codegen done, not run on the GPU",
                        },
                        (major, minor) => Support::Unsupported(format!("GFX {major}.{minor}: detour targets RDNA2 (10.3) only")),
                    }
                }
                Err(e) => Support::Unsupported(format!("opening it failed: {e}")),
            }
        }
        #[cfg(not(feature = "rdna2"))]
        {
            let _ = verbose;
            Support::NotBuilt { feature: "rdna2" }
        }
    }

    fn asahi(&self, verbose: bool) -> Support {
        #[cfg(feature = "agx-g13")]
        {
            use detour::asahi::{Asahi, Isa};
            use detour::base::Gpu;
            match Asahi::open(&self.path) {
                Ok(asahi) => {
                    let info = asahi.driver_info();
                    if verbose {
                        println!("{info:#?}");
                    }
                    let model = format!("{} ({:?})", info.name(), info.chip);
                    match info.chip.isa() {
                        Isa::Agx2 => Support::Native { model, state: "compute codegen done, not run on the GPU" },
                        Isa::Apple9 => Support::Native { model, state: "Apple9: identified only" },
                    }
                }
                Err(e) => Support::Unsupported(format!("{e}")),
            }
        }
        #[cfg(not(feature = "agx-g13"))]
        {
            let _ = verbose;
            Support::NotBuilt { feature: "agx-g13" }
        }
    }

    fn print(&self, verbose: bool) {
        let ids = match self.pci {
            Some((v, d)) => format!("{v:04x}:{d:04x}"),
            None => self.compatible.join(" "),
        };
        println!("{}  {}  {ids}", self.path.display(), self.driver_name);
        match self.support(verbose) {
            Support::Native { model, state } => println!("  detour: {model}: {state}"),
            Support::NotBuilt { feature } => println!("  detour: has a driver, not in this build (feature `{feature}`)"),
            Support::Unsupported(why) => println!("  detour: no ({why})"),
        }
    }
}

impl Devices {
    pub fn run(&self) -> io::Result<()> {
        let mut nodes: Vec<PathBuf> = fs::read_dir("/dev/dri")?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("renderD")))
            .collect();
        nodes.sort();
        if nodes.is_empty() {
            println!("no DRM render nodes in /dev/dri");
        }
        for path in nodes {
            Node::read(path).print(self.verbose);
        }
        let fallback = if cfg!(feature = "fallback-vulkan") { "built in" } else { "off: what no driver does panics" };
        println!("fallback (vello on wgpu on Vulkan): {fallback}");
        Ok(())
    }
}
