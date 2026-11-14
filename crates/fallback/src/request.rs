//! The device a page rasterizer and WebGPU share, asked for in tiers (moved back from acde's acde-gpu): all of the
//! adapter's WebGPU features, then none, then downlevel limits, so a driver that rejects an optional feature still opens.

#[derive(Clone, Debug, PartialEq)]
pub struct DeviceRequest {
    pub tier: &'static str,
    pub features: wgpu::Features,
    pub limits: wgpu::Limits,
}

impl DeviceRequest {
    pub fn tiers(features: wgpu::Features, limits: wgpu::Limits) -> Vec<Self> {
        let fallback = wgpu::Limits::downlevel_defaults().using_resolution(limits.clone());
        let tiers = [
            Self { tier: "WebGPU features", features, limits: limits.clone() },
            Self { tier: "no optional features", features: wgpu::Features::empty(), limits },
            Self { tier: "downlevel limits", features: wgpu::Features::empty(), limits: fallback },
        ];
        let mut unique: Vec<Self> = Vec::new();
        for tier in tiers {
            if !unique.iter().any(|seen| seen.features == tier.features && seen.limits == tier.limits) {
                unique.push(tier);
            }
        }
        unique
    }

    pub fn open<T>(label: &str, tiers: &[Self], mut open: impl FnMut(&Self) -> Result<T, String>) -> Result<(T, &'static str), String> {
        let mut errors = Vec::new();
        for request in tiers {
            match open(request) {
                Ok(device) => {
                    log::info!("{label}: GPU device opened with {}", request.tier);
                    return Ok((device, request.tier));
                }
                Err(e) => {
                    log::warn!("{label}: requesting the GPU device with {} ({:?}) failed: {e}", request.tier, request.features);
                    errors.push(format!("{}: {e}", request.tier));
                }
            }
        }
        Err(errors.join("; "))
    }

    pub fn descriptor<'a>(&self, label: &'a str) -> wgpu::DeviceDescriptor<'a> {
        wgpu::DeviceDescriptor { label: Some(label), required_features: self.features, required_limits: self.limits.clone(), ..Default::default() }
    }
}
