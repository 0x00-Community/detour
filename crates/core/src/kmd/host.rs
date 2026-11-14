//! The CPU side: what it takes for the GPU to see what the CPU wrote, per CPU architecture.

/// ⭐ The host CPU.
pub trait Host {
    /// Writes `bytes` back from the CPU caches (for buffers the GPU does not snoop).
    fn flush(bytes: &[u8]);
    /// Orders the CPU's earlier stores (write-combined ones included) before its later ones.
    fn fence();
}

/// One `Host` per architecture: the line size it flushes by, how to flush one line, how to fence.
macro_rules! hosts {
    ($($(#[$m:meta])* $name:ident: $arch:literal, line $line:literal, flush |$p:ident| $flush:expr, fence $fence:expr;)*) => {$(
        $(#[$m])*
        #[cfg(target_arch = $arch)]
        pub struct $name;

        #[cfg(target_arch = $arch)]
        impl Host for $name {
            fn flush(bytes: &[u8]) {
                for line in bytes.chunks($line) {
                    let $p = line.as_ptr();
                    // SAFETY: `$p` points into `bytes`.
                    unsafe { $flush };
                }
                Self::fence();
            }

            fn fence() {
                // SAFETY: a plain barrier.
                unsafe { $fence };
            }
        }
    )*};
}

hosts! {
    /// x86-64: `clflush` per line, `mfence` (orders WC stores too).
    X86_64: "x86_64", line 64, flush |p| core::arch::x86_64::_mm_clflush(p), fence core::arch::x86_64::_mm_mfence();
    /// AArch64 (Apple M1 ..): `dc civac` per line to the point of coherency, `dsb sy`. A 64-byte stride flushes every
    /// line of any core whose CTR_EL0.DminLine is 64 bytes or more (Apple's are 128: some lines twice, none missed).
    Aarch64: "aarch64", line 64, flush |p| core::arch::asm!("dc civac, {}", in(reg) p, options(nostack, preserves_flags)),
        fence core::arch::asm!("dsb sy", options(nostack, preserves_flags));
}

/// The CPU this is built for.
#[cfg(target_arch = "x86_64")]
pub type Native = X86_64;
#[cfg(target_arch = "aarch64")]
pub type Native = Aarch64;
