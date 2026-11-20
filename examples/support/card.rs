//! The card an example draws on (the first there is), and its own tiled layout: Tile4 on Xe3, linear on RDNA2

use detour::{Card, Tiling};

thread_local! {
    static CARD: &'static Card = Box::leak(Box::new(Card::open_first().expect("an Xe3 or RDNA2 render node")));
}

/// The first card detour draws on
pub fn open() -> &'static Card {
    CARD.with(|c| *c)
}

/// The card's own tiled layout
#[allow(dead_code, reason = "not every example tiles")]
pub fn tiled() -> Tiling {
    match open() {
        Card::Xe(_) => Tiling::Tile4,
        Card::Amdgpu(_) => Tiling::Linear,
    }
}
