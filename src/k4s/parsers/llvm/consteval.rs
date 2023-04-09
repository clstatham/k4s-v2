use llvm_ir::{
    constant::Float,
    types::{FPType, Typed, Types},
    Constant, Name, Type,
};

use crate::k4s::{contexts::llvm::FunctionContext, InstrSize, Token};

use super::Ssa;

impl Ssa {
    pub fn parse_const(con: &Constant, name: Name, types: &Types) -> Self {
        let ty = con.get_type(types);
        match con {
            Constant::Int { bits, value } => Self::new(
                name,
                ty,
                Token::from_integer_size(*value, InstrSize::from_integer_bits(*bits).unwrap())
                    .unwrap(),
            ),
            Constant::Float(float) => {
                let tok = match float {
                    Float::Single(val) => Token::F32(*val),
                    Float::Double(val) => Token::F64(*val),
                    _ => todo!("{:?}", con),
                };
                Self::new(name, ty, tok)
            }

            _ => todo!("{:?}", con),
        }
    }
}

pub trait NameExt {
    fn strip_prefix(&self) -> String;
}

impl NameExt for Name {
    fn strip_prefix(&self) -> String {
        self.to_string()
            .strip_prefix('%')
            .unwrap_or(&self.to_string())
            .to_string()
    }
}

pub trait TypeExt {
    fn total_size_in_bytes(&self) -> usize;
}

impl TypeExt for Type {
    fn total_size_in_bytes(&self) -> usize {
        match self {
            Type::IntegerType { bits } => 1.max(*bits as usize / 8),
            Type::FPType(float) => match float {
                FPType::Single => 4,
                FPType::Double => 8,
                _ => todo!("{:?}", self),
            },
            Type::VoidType => 0,
            Type::PointerType { .. } => 8,
            Type::FuncType { .. } => 8,
            Type::LabelType => 8,
            Type::ArrayType {
                element_type,
                num_elements,
            }
            | Type::VectorType {
                element_type,
                num_elements,
                ..
            } => element_type.as_ref().total_size_in_bytes() * *num_elements,
            Type::StructType {
                element_types,
                is_packed,
            } => {
                assert!(*is_packed, "only packed structs are supported currently");
                element_types
                    .iter()
                    .map(|ty| ty.as_ref().total_size_in_bytes())
                    .sum()
            }
            _ => todo!("{:?}", self),
        }
    }
}
