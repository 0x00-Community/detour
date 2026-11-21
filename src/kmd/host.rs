pub trait Host {
    fn flush(bytes: &[u8]);
    fn fence();
}

macro_rules! hosts {
    ($($(#[$m:meta])* $name:ident: $arch:literal, line $line:expr, flush |$p:ident| $flush:expr, fence $fence:expr;)*) => {$(
        $(#[$m])*
        #[cfg(target_arch = $arch)]
        pub struct $name;

        #[cfg(target_arch = $arch)]
        impl Host for $name {
            fn flush(bytes: &[u8]) {
                for line in bytes.chunks($line) {
                    let $p = line.as_ptr();
                    unsafe { $flush };
                }
                Self::fence();
            }

            fn fence() {
                unsafe { $fence };
            }
        }
    )*};
}

hosts! {
    X86_64: "x86_64", line 64, flush |p| core::arch::x86_64::_mm_clflush(p), fence core::arch::x86_64::_mm_mfence();
    Aarch64: "aarch64", line Aarch64::line(), flush |p| core::arch::asm!("dc civac, {p}", p = in(reg) p, options(nostack, preserves_flags)), fence core::arch::asm!("dsb sy", options(nostack, preserves_flags));
}

/// CTR_EL0.DminLine: log2 of the smallest data cache line, in 4-byte words
#[cfg(target_arch = "aarch64")]
const CTR_DMINLINE_SHIFT: u64 = 16;
#[cfg(target_arch = "aarch64")]
const CTR_DMINLINE_MASK: u64 = 0xf;
#[cfg(target_arch = "aarch64")]
const CTR_WORD: usize = 4;

#[cfg(target_arch = "aarch64")]
impl Aarch64 {
    /// The smallest data cache line, read from CTR_EL0 (Linux lets user space read it)
    fn line() -> usize {
        let ctr: u64;
        unsafe { core::arch::asm!("mrs {ctr}, ctr_el0", ctr = out(reg) ctr, options(nomem, nostack, preserves_flags)) };
        CTR_WORD << (ctr >> CTR_DMINLINE_SHIFT & CTR_DMINLINE_MASK)
    }
}

#[cfg(target_arch = "x86_64")]
pub type Native = X86_64;
#[cfg(target_arch = "aarch64")]
pub type Native = Aarch64;
