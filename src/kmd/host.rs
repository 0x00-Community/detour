pub trait Host {
    fn flush(bytes: &[u8]);
    fn fence();
}

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
}

#[cfg(target_arch = "x86_64")]
pub type Native = X86_64;
