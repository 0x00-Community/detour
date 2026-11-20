#[allow(unused_imports, reason = "the alloc prelude")]
use alloc::{borrow::ToOwned, boxed::Box, format, string::{String, ToString}, vec, vec::Vec};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Elem {
    UB = 0b00000,
    UW = 0b00001,
    UD = 0b00010,
    UQ = 0b00011,
    B = 0b00100,
    W = 0b00101,
    D = 0b00110,
    Q = 0b00111,
    HF = 0b01001,
    F = 0b01010,
    DF = 0b01011,
    BF = 0b01101,
    UV = 0b10001,
    V = 0b10101,
    VF = 0b11010,
}

impl Elem {
    pub const fn size(self) -> u32 {
        match self {
            Elem::UV | Elem::V | Elem::VF => 4,
            t => 1 << (t as u8 & 3),
        }
    }

    pub const fn is_float(self) -> bool {
        (self as u8 & 0b01000) != 0 && (self as u8 & 0b10000) == 0
    }

    pub const fn code(self) -> u64 {
        match self {
            Elem::UV | Elem::V | Elem::VF => (self as u8 & !(0b10000 | 0b00011)) as u64,
            t => (t as u8 & 0b01111) as u64,
        }
    }

    pub const fn code3(self) -> u64 {
        match self {
            Elem::UB => 0b000,
            Elem::UW => 0b001,
            Elem::UD => 0b010,
            Elem::UQ => 0b011,
            Elem::B => 0b100,
            Elem::W => 0b101,
            Elem::D => 0b110,
            Elem::Q => 0b111,
            Elem::HF => 0b001,
            Elem::F => 0b010,
            Elem::DF => 0b011,
            Elem::BF => 0b101,
            Elem::UV | Elem::V | Elem::VF => panic!("vector immediate in a three source instruction"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum File {
    Arf,
    Grf,
    Imm,
}

pub struct Arf;

impl Arf {
    pub const NULL: u8 = 0x00;
    pub const ADDRESS: u8 = 0x10;
    pub const ACCUMULATOR: u8 = 0x20;
    pub const FLAG: u8 = 0x30;
    pub const MASK: u8 = 0x40;
    pub const SCALAR: u8 = 0x60;
    pub const STATE: u8 = 0x70;
    pub const CONTROL: u8 = 0x80;
    pub const NOTIFICATION_COUNT: u8 = 0x90;
    pub const IP: u8 = 0xa0;
    pub const TDR: u8 = 0xb0;
    pub const TIMESTAMP: u8 = 0xc0;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    pub vstride: u8,
    pub width: u8,
    pub hstride: u8,
}

impl Region {
    pub const fn new(vstride: u8, width: u8, hstride: u8) -> Region {
        Region { vstride, width, hstride }
    }
    pub const SCALAR: Region = Region::new(0, 1, 0);

    pub const fn step(self) -> u8 {
        if self.width == 1 { self.vstride } else { self.hstride }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Operand {
    pub file: File,
    pub ty: Elem,
    pub nr: u8,
    pub subnr: u8,
    pub region: Region,
    pub negate: bool,
    pub abs: bool,
    pub imm: u64,
}

impl Operand {
    pub const fn grf(nr: u8, ty: Elem) -> Operand {
        Operand {
            file: File::Grf,
            ty,
            nr,
            subnr: 0,
            region: Region::new(1, 1, 0),
            negate: false,
            abs: false,
            imm: 0,
        }
    }

    pub const fn arf(nr: u8, ty: Elem) -> Operand {
        Operand { file: File::Arf, ..Operand::grf(nr, ty) }
    }

    pub const fn null() -> Operand {
        Operand::arf(Arf::NULL, Elem::UD)
    }

    pub const fn imm(ty: Elem, bits: u64) -> Operand {
        Operand { file: File::Imm, imm: bits, region: Region::SCALAR, ..Operand::grf(0, ty) }
    }

    pub const fn sub(mut self, subnr: u8) -> Operand {
        self.subnr = subnr;
        self
    }

    pub const fn ty(mut self, ty: Elem) -> Operand {
        self.ty = ty;
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    One,
    Two,
    Three,
    Send,
    Branch1,
    Branch2,
    Nop,
}

macro_rules! opcodes {
    ($($name:ident = $hw:expr, $format:ident;)*) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum Opcode { $($name),* }

        impl Opcode {
            pub const fn hw(self) -> u64 {
                match self { $(Opcode::$name => $hw),* }
            }
            pub const fn format(self) -> Format {
                match self { $(Opcode::$name => Format::$format),* }
            }

            pub const ALL: &[Opcode] = &[$(Opcode::$name),*];

            pub fn mnemonic(self) -> String {
                #[allow(unreachable_patterns, reason = "NopOp is spelled nop")]
                match self {
                    Opcode::NopOp => "nop".into(),
                    $(Opcode::$name => stringify!($name).to_ascii_lowercase()),*
                }
            }
        }
    };
}

opcodes! {
    Illegal = 0, Nop;
    Bfrev = 119, One;
    Cbit = 77, One;
    Fbh = 75, One;
    Fbl = 76, One;
    Frc = 67, One;
    Lzd = 74, One;
    Mov = 97, One;
    Not = 100, One;
    Rndd = 69, One;
    Rnde = 70, One;
    Rndu = 68, One;
    Rndz = 71, One;
    Sync = 1, One;
    Wait = 48, One;
    Add = 64, Two;
    Addc = 78, Two;
    And = 101, Two;
    Asr = 108, Two;
    Avg = 66, Two;
    Bfi1 = 121, Two;
    Cmp = 112, Two;
    Cmpn = 113, Two;
    Mac = 72, Two;
    Mach = 73, Two;
    Macl = 83, Two;
    Math = 56, Two;
    Movi = 99, Two;
    Mul = 65, Two;
    Or = 102, Two;
    Rol = 111, Two;
    Ror = 110, Two;
    Sel = 98, Two;
    Shl = 105, Two;
    Shr = 104, Two;
    Smov = 106, Two;
    Srnd = 84, Two;
    Subb = 79, Two;
    Xor = 103, Two;
    Add3 = 82, Three;
    Bfe = 120, Three;
    Bfi2 = 122, Three;
    Bfn = 107, Three;
    Csel = 114, Three;
    Dp4a = 88, Three;
    Mad = 91, Three;
    Madm = 93, Three;
    Send = 49, Send;
    Sendc = 50, Send;
    Brd = 33, Branch1;
    Call = 44, Branch1;
    Calla = 43, Branch1;
    Endif = 37, Branch1;
    Jmpi = 32, Branch1;
    Join = 47, Branch1;
    Ret = 45, Branch1;
    While = 39, Branch1;
    Brc = 35, Branch2;
    Break = 40, Branch2;
    Continue = 41, Branch2;
    Else = 36, Branch2;
    Goto = 46, Branch2;
    Halt = 42, Branch2;
    If = 34, Branch2;
    NopOp = 96, Nop;
}

impl Opcode {
    pub fn has_dst(self) -> bool {
        !matches!(self.format(), Format::Branch1 | Format::Branch2 | Format::Nop) && !matches!(self, Opcode::Sync | Opcode::Smov)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum Cond {
    #[default]
    None = 0,
    Z = 1,
    Nz = 2,
    Gt = 3,
    Ge = 4,
    Lt = 5,
    Le = 6,
    Ov = 8,
    Un = 9,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum Pred {
    #[default]
    None = 0,
    Normal = 1,
    Any = 2,
    All = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MathFn {
    Inv = 1,
    Log = 2,
    Exp = 3,
    Sqrt = 4,
    Rsq = 5,
    Sin = 6,
    Cos = 7,
    IntDivBoth = 11,
    IntDivQuotient = 12,
    IntDivRemainder = 13,
    Invm = 14,
    Rsqrtm = 15,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SyncFn {
    Nop = 0x0,
    AllRd = 0x2,
    AllWr = 0x3,
    Fence = 0xd,
    Bar = 0xe,
    Host = 0xf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Pipe {
    #[default]
    None,
    Float,
    Int,
    Long,
    Math,
    Scalar,
    All,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Swsb {
    pub regdist: u8,
    pub pipe: Pipe,
    pub sbid: u8,
    pub mode: SbidMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SbidMode {
    #[default]
    Null,
    Src,
    Dst,
    Set,
}

impl Swsb {
    pub const NONE: Swsb = Swsb { regdist: 0, pipe: Pipe::None, sbid: 0, mode: SbidMode::Null };

    pub const fn regdist(pipe: Pipe, n: u8) -> Swsb {
        Swsb { regdist: n, pipe, sbid: 0, mode: SbidMode::Null }
    }

    pub fn encode(self, op: Opcode) -> u64 {
        let pipe = match self.pipe {
            Pipe::Float => 0x10,
            Pipe::Int => 0x18,
            Pipe::Long => 0x20,
            Pipe::Math => 0x28,
            Pipe::Scalar => 0x30,
            Pipe::All => 0x08,
            Pipe::None => 0,
        };
        let (rd, sbid) = (self.regdist as u64, self.sbid as u64);
        match self.mode {
            SbidMode::Null => pipe | rd,
            _ if rd != 0 => {
                let mode = match self.mode {
                    SbidMode::Set => {
                        debug_assert!(matches!(op, Opcode::Send | Opcode::Sendc));
                        match self.pipe {
                            Pipe::Int => 0b11,
                            Pipe::Float => 0b10,
                            _ => 0b01,
                        }
                    }
                    _ if self.pipe == Pipe::All => 0b11,
                    SbidMode::Src => 0b10,
                    _ => 0b01,
                };
                mode << 8 | rd << 5 | sbid
            }
            SbidMode::Set => sbid | 0xc0,
            SbidMode::Dst => sbid | 0x80,
            SbidMode::Src => sbid | 0xa0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct SendInfo {
    pub sfid: u8,
    pub desc: u32,
    pub ex_desc: u32,
    pub src1_len: u8,
    pub eot: bool,
    pub ex_bso: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Inst {
    pub op: Opcode,
    pub exec_size: u8,
    pub chan_offset: u8,
    pub cond: Cond,
    pub pred: Pred,
    pub pred_inv: bool,
    pub flag: (u8, u8),
    pub saturate: bool,
    pub no_mask: bool,
    pub atomic: bool,
    pub swsb: Swsb,
    pub dst: Operand,
    pub src: [Operand; 3],
    pub math: Option<MathFn>,
    pub sync: SyncFn,
    pub bfn_ctrl: u8,
    pub send: SendInfo,
    pub jip: i32,
    pub uip: i32,
    pub branch_ctrl: bool,
}

impl Inst {
    pub const fn new(op: Opcode, exec_size: u8) -> Inst {
        Inst {
            op,
            exec_size,
            chan_offset: 0,
            cond: Cond::None,
            pred: Pred::None,
            pred_inv: false,
            flag: (0, 0),
            saturate: false,
            no_mask: false,
            atomic: false,
            swsb: Swsb::NONE,
            dst: Operand::null(),
            src: [Operand::null(); 3],
            math: None,
            sync: SyncFn::Nop,
            bfn_ctrl: 0,
            send: SendInfo { sfid: 0, desc: 0, ex_desc: 0, src1_len: 0, eot: false, ex_bso: false },
            jip: 0,
            uip: 0,
            branch_ctrl: false,
        }
    }
}
