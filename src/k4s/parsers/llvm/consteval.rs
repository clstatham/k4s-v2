use llvm_ir::{
    constant::Float,
    types::{FPType, NamedStructDef, Typed, Types},
    Constant, ConstantRef, Name, Type,
};
use rustc_hash::FxHashMap;

use crate::k4s::{Data, InstrSize, Label, Token};

use super::Ssa;

impl Ssa {
    pub fn parse_const(
        con: &ConstantRef,
        name: Name,
        types: &Types,
        globals: &FxHashMap<Name, Ssa>,
    ) -> Self {
        let ty = con.get_type(types);
        match con.as_ref() {
            Constant::Int { value, .. } => Self::new(
                name,
                ty.to_owned(),
                Token::from_integer_size(*value, ty.instr_size(types)).unwrap(),
                None,
            ),
            Constant::Float(float) => {
                let tok = match float {
                    Float::Single(val) => Token::F32(*val),
                    Float::Double(val) => Token::F64(*val),
                    _ => todo!("{:?}", con),
                };
                Self::new(name, ty, tok, None)
            }
            Constant::GlobalReference {
                name: ref_name,
                ty: ref_ty,
            } => Self::new(
                name,
                Type::PointerType {
                    pointee_type: ref_ty.to_owned(),
                    addr_space: 0,
                }
                .get_type(types),
                // Token::Data(Data {
                globals
                    .get(ref_name)
                    .map(|ssa| ssa.storage().to_owned())
                    .unwrap_or_else(|| Token::Label(Label::new_unlinked(ref_name.strip_prefix()))),
                // Token::Label(Label::new_unlinked(ref_name.strip_prefix())),
                // align: 1,
                // data: Vec::new(),
                // }),
                None,
            ),
            Constant::Undef(ty) => Self::new(
                name,
                ty.to_owned(),
                Token::I64(0), // todo?
                None,
            ),
            Constant::Null(ty) => Self::new(name, ty.to_owned(), Token::I64(0), None),
            Constant::Struct {
                name: str_name,
                values: _,
                is_packed: _,
            } => {
                // todo
                Self::new(
                    name.to_owned(),
                    ty,
                    Token::Data(Data {
                        label: Label::new_unlinked(
                            str_name
                                .as_ref()
                                .unwrap_or(&name.strip_prefix())
                                .to_string(),
                        ),
                        align: 0,
                        data: Vec::new(),
                    }),
                    Some(con.to_owned()),
                )
            }
            Constant::Array {
                element_type,
                elements,
            } => {
                let storage = match element_type.as_ref() {
                    Type::IntegerType { bits: 8 } => Token::Data(Data {
                        label: Label::new_unlinked(name.strip_prefix()),
                        align: 1,
                        data: elements
                            .iter()
                            .enumerate()
                            .map(|(i, elem)| {
                                let name = format!("{}_elem{}", name.to_owned().strip_prefix(), i);
                                Self::parse_const(elem, name.into(), types, globals)
                                    .storage()
                                    .as_integer()
                                    .unwrap()
                            })
                            .collect(),
                    }),
                    _ => todo!(),
                };
                Self::new(name, ty, storage, None) // todo: set agg_const when doing arrays of non-u8's
            }
            Constant::AggregateZero(agg) => Self::new(
                name.to_owned(),
                ty,
                Token::Data(Data {
                    label: Label::new_unlinked(name.strip_prefix()),
                    align: 1,
                    data: vec![0u8; agg.total_size_in_bytes(types)],
                }),
                None,
            ),
            Constant::BitCast(cast) => {
                let op = Self::parse_const(&cast.operand, name.to_owned(), types, globals);
                Self::new(name, cast.to_type.to_owned(), op.storage().to_owned(), None)
            }
            Constant::GetElementPtr(gep) => {
                let addr = Self::parse_const(&gep.address, name.to_owned(), types, globals);
                let indices = gep
                    .indices
                    .iter()
                    .map(|idx| Self::parse_const(idx, name.to_owned(), types, globals))
                    .collect::<Vec<_>>();
                let mut current_type = addr.ty().as_ref().to_owned();
                let mut total = 0;
                for idx in indices.iter() {
                    let idx = idx.storage().as_integer::<u64>().unwrap();
                    if let Type::NamedStructType { name: struc_name } = current_type.clone() {
                        if let NamedStructDef::Defined(struc_ty) =
                            types.named_struct_def(&struc_name).unwrap()
                        {
                            current_type = struc_ty.as_ref().to_owned();
                        } else {
                            todo!("opaque structs")
                        }
                    }
                    match current_type.clone() {
                        Type::StructType {
                            element_types,
                            is_packed: _,
                        } => {
                            if idx > 0 {
                                total += element_types[..idx as usize]
                                    .iter()
                                    .map(|elem| elem.total_size_in_bytes(types) as u64)
                                    .sum::<u64>();
                            }
                            current_type = element_types[idx as usize].as_ref().to_owned();
                        }
                        Type::PointerType {
                            pointee_type: element_type,
                            ..
                        }
                        | Type::ArrayType { element_type, .. }
                        | Type::VectorType { element_type, .. } => {
                            current_type = element_type.as_ref().to_owned();
                            if idx > 0 {
                                total += current_type.total_size_in_bytes(types) as u64 * idx;
                            }
                        }
                        t => todo!("{:?}", t),
                    }
                }

                let addr_label = if let Token::Data(data) = addr.storage() {
                    data.label.clone()
                } else if let Token::Label(lab) = addr.storage() {
                    lab.clone()
                } else {
                    unreachable!()
                };
                Self::new(
                    name,
                    Type::PointerType {
                        pointee_type: current_type.get_type(types),
                        addr_space: 0,
                    }
                    .get_type(types),
                    Token::LabelOffset(total as i64, addr_label),
                    None,
                )
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
        self.to_string().strip_prefix('%').unwrap().to_string()
    }
}

pub trait TypeExt {
    fn total_size_in_bytes(&self, types: &Types) -> usize;
    fn instr_size(&self, types: &Types) -> InstrSize;
}

impl TypeExt for Type {
    fn total_size_in_bytes(&self, types: &Types) -> usize {
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
            } => element_type.as_ref().total_size_in_bytes(types) * *num_elements,
            Type::StructType {
                element_types,
                is_packed: _,
            } => {
                // assert!(*is_packed, "only packed structs are supported currently"); // todo: enforce this or implement struct padding
                element_types
                    .iter()
                    .map(|ty| ty.as_ref().total_size_in_bytes(types))
                    .sum()
            }
            Type::NamedStructType { name } => {
                let struc = types.named_struct_def(name).unwrap();
                if let NamedStructDef::Defined(ty) = struc {
                    ty.total_size_in_bytes(types)
                } else {
                    todo!("opaque structs")
                }
            }
            _ => todo!("{:?}", self),
        }
    }

    fn instr_size(&self, types: &Types) -> InstrSize {
        if let Type::FPType(precision) = self {
            match precision {
                FPType::Single => InstrSize::F32,
                FPType::Double => InstrSize::F64,
                _ => todo!("{:?}", precision),
            }
        } else {
            InstrSize::from_integer_bits(self.get_type(types).total_size_in_bytes(types) as u32 * 8)
                .unwrap_or_else(|| match self.get_type(types).as_ref() {
                    Type::ArrayType { .. } => InstrSize::I64,
                    Type::StructType { .. } => InstrSize::I64,
                    Type::IntegerType { bits: 24 } => InstrSize::I32,
                    Type::IntegerType { bits: 48 } => InstrSize::I64,
                    Type::IntegerType { bits: 56 } => InstrSize::I64,
                    Type::IntegerType { bits: 96 } => InstrSize::I128,

                    ty => todo!("{:?}", ty),
                })
        }
    }
}
