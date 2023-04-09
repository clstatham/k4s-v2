use anyhow::{Context, Error, Result};
use nom::bytes::complete::{escaped, escaped_transform, take, tag, is_a};
use nom::character::complete::alphanumeric1;
use nom::combinator::map;
use nom::multi::{many0, many1};
use nom::sequence::{tuple, preceded};
use nom::branch::alt;
use nom::{IResult, AsBytes};
use rustc_hash::FxHashMap;

use crate::k4s::{
    parsers::{
        asm::{data, header, label, literal, opcode, size, token},
        machine::tags,
    },
    Data, Instr, InstrSize, Label, Linkage, Primitive, Token,
};

#[derive(Debug, Clone)]
pub enum Header {
    Entry(u64),
    Include(String),
}

#[derive(Debug, Clone)]
pub enum ParsedLine {
    Comment,
    Header(Header),
    Label(Label),
    Data(Data),
    Instr(Instr),
}

impl ParsedLine {
    pub fn parse(line: &str) -> anyhow::Result<(&str, Self)> {
        let line = line.trim();
        if line.starts_with(';') {
            return Ok((line, ParsedLine::Comment));
        }
        if let Ok((rest, header)) = header(line) {
            let rest = rest.trim();
            let header = match header {
                "!ent" => {
                    if rest.starts_with("0x") {
                        let rest = rest.strip_prefix("0x").unwrap_or(rest);
                        Header::Entry(u64::from_str_radix(rest, 16).unwrap())
                    } else {
                        return Err(Error::msg("Expected hex number after `!ent` tag"));
                    }
                }
                _ => return Err(Error::msg(format!("Invalid header tag: {header}"))),
            };
            Ok(("", ParsedLine::Header(header)))
        } else if let Ok((rest, label)) = label(line) {
            if let Token::Label(label) = label {
                Ok((rest, ParsedLine::Label(label)))
            } else {
                unreachable!()
            }
        } else if let Ok((rest, data)) = data(line) {
            if let Token::Data(data) = data {
                Ok((rest, ParsedLine::Data(data)))
            } else {
                unreachable!()
            }
        } else {
            let (rest, opcode) = opcode(line)
                .map_err(|e| e.to_owned())
                .context("failed to parse opcode")?;
            if opcode.n_args() > 0 {
                let (rest, size) = size(rest.trim())
                    .map_err(|e| e.to_owned())
                    .context("failed to parse instruction size")?;
                let (rest, arg0, arg1) = if opcode.n_args() > 0 {
                    let (rest, arg0) = token(size, rest.trim())
                        .map_err(|e| e.to_owned())
                        .context("failed to parse first argument")?;
                    let (rest, arg1) = if opcode.n_args() > 1 {
                        let arg1 = token(size, rest.trim())
                            .map_err(|e| e.to_owned())
                            .context("failed to parse second argument")?;
                        (arg1.0, Some(arg1.1))
                    } else {
                        (rest, None)
                    };
                    (rest, Some(arg0), arg1)
                } else {
                    (rest, None, None)
                };
                Ok((
                    rest,
                    ParsedLine::Instr(Instr {
                        opcode,
                        size,
                        arg0,
                        arg1,
                    }),
                ))
            } else {
                Ok((
                    rest,
                    ParsedLine::Instr(Instr {
                        opcode,
                        size: InstrSize::Unsized,
                        arg0: None,
                        arg1: None,
                    }),
                ))
            }
        }
    }
}

pub struct AssemblyContext {
    pub lines: Vec<ParsedLine>,
    pub linked_refs: FxHashMap<Label, u64>, // (label, addr found at)
    pub unlinked_refs: FxHashMap<Label, u64>, // (label, addr found at)
    pub input: String,
    pub pc: u64,
    pub entry_point: u64,
    pub output: Vec<u8>,
}

impl AssemblyContext {
    pub fn new(input: String) -> Self {
        Self {
            lines: Vec::default(),
            linked_refs: FxHashMap::default(),
            unlinked_refs: FxHashMap::default(),
            input,
            pc: 0,
            entry_point: 0,
            output: Vec::new(),
        }
    }

    pub fn push_program_bytes(&mut self, bytes: &[u8]) {
        self.output.extend_from_slice(bytes);
        self.pc += bytes.len() as u64;
    }

    pub fn assemble(&mut self) -> Result<Vec<u8>> {
        let mut in_header = true;
        for (line_no, line) in self.input.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let line_no = line_no + 1;
            let (junk, mut parsed_line) = ParsedLine::parse(line)
                .context(format!("Error while parsing line {}:\n{}", line_no, line))?;
            if let ParsedLine::Data(ref mut data) = parsed_line {
                // parse the junk as data
                let string = junk.trim();
                if let Ok((_junk, lit)) = literal(InstrSize::I64, string) {
                    let lit = lit.as_integer::<u64>().unwrap();
                    data.data = lit.to_bytes().to_vec();
                } else if let Some(string) = string.strip_prefix('"') {
                    if let Some(string) = string.strip_suffix('"') {
                        let bytes = string.to_owned().into_bytes();
                        let mut cursor = 0;
                        while cursor < bytes.len() {
                            if &bytes[cursor..bytes.len().min(cursor + 2)] == br"\x" {
                                let d = u8::from_str_radix(
                                    std::str::from_utf8(&bytes[cursor + 2..cursor + 4]).unwrap(),
                                    16,
                                )
                                .unwrap();
                                data.data.push(d);
                                cursor += 4;
                            } else {
                                data.data.push(bytes[cursor]);
                                cursor += 1;
                            }
                        }
                    } else {
                        return Err(Error::msg(format!("Expected closing `\"` after opening `\"` in data on line {line_no}")))
                    }
                } else {
                    return Err(Error::msg(format!("Invalid data on line {line_no}")))
                }
            } else if !junk.is_empty() && !junk.trim().starts_with(';') {
                eprintln!("Warning: Ignoring junk after line {line_no}: {junk}");
            }
            if let ParsedLine::Header(ref header) = parsed_line {
                if !in_header {
                    return Err(Error::msg(format!(
                        "Found header line outside of header on line {line_no}: {line}"
                    )));
                }
                match header {
                    Header::Entry(pc) => {
                        self.pc = *pc;
                        self.entry_point = *pc;
                    }
                    Header::Include(_path) => todo!(),
                }
            } else {
                in_header = false;
            }
            self.lines.push(parsed_line);
        }

        let mut lines = self.lines.clone();
        for line in lines.iter_mut() {
            match line {
                ParsedLine::Comment => {}
                ParsedLine::Header(_) => {}
                ParsedLine::Label(lab) => {
                    if let Some(_old) = self.linked_refs.insert(lab.to_owned(), self.pc) {
                        return Err(Error::msg(format!("Found duplicate label: {}", lab.name)));
                    }
                }
                ParsedLine::Data(dat) => {
                    if let Some(_old) = self.linked_refs.insert(dat.label.to_owned(), self.pc) {
                        return Err(Error::msg(format!(
                            "Found duplicate data label: {}",
                            dat.label.name
                        )));
                    }
                    self.push_program_bytes(&dat.data);
                }
                ParsedLine::Instr(ins) => {
                    ins.assemble(self)?;
                }
            }
        }

        // one more linking phase, for the forward decls
        for line in lines.iter_mut() {
            if let ParsedLine::Instr(ins) = line {
                if let Some(Token::Label(ref mut lab)) = ins.arg0 {
                    if lab.linkage == Linkage::NeedsLinking {
                        if let Some(link_location) = self.linked_refs.get(lab) {
                            let ref_loc = self.unlinked_refs.remove(lab).unwrap_or_else(|| {
                                panic!(
                                    "Label {} needed linking, but wasn't in the unlinked refs",
                                    lab.name
                                )
                            }) as usize;
                            lab.linkage = Linkage::Linked(*link_location);
                            self.output[ref_loc - self.entry_point as usize
                                ..ref_loc - self.entry_point as usize + 8]
                                .copy_from_slice(&link_location.to_bytes());
                        } else {
                            return Err(Error::msg(format!(
                                "Undefined reference to label {}",
                                lab.name
                            )));
                        }
                    }
                }
                if let Some(Token::Label(ref mut lab)) = ins.arg1 {
                    if lab.linkage == Linkage::NeedsLinking {
                        if let Some(link_location) = self.linked_refs.get(lab) {
                            let ref_loc = self.unlinked_refs.remove(lab).unwrap_or_else(|| {
                                panic!(
                                    "Label {} needed linking, but wasn't in the unlinked refs",
                                    lab.name
                                )
                            }) as usize;
                            lab.linkage = Linkage::Linked(*link_location);
                            self.output[ref_loc - self.entry_point as usize
                                ..ref_loc - self.entry_point as usize + 8]
                                .copy_from_slice(&link_location.to_bytes());
                        } else {
                            return Err(Error::msg(format!(
                                "Undefined reference to label {}",
                                lab.name
                            )));
                        }
                    }
                }
            }
        }

        if !self.unlinked_refs.is_empty() {
            for lab in self.unlinked_refs.keys() {
                eprintln!(
                    "Undefined reference to label {} after stage 2 of linking",
                    lab.name
                );
            }
            return Err(Error::msg(
                "Undefined references to labels after stage 2 of linking",
            ));
        }

        let mut final_out = Vec::new();
        final_out.extend_from_slice(tags::HEADER_MAGIC);
        final_out.extend_from_slice(tags::HEADER_ENTRY_POINT);
        final_out.extend_from_slice(&self.entry_point.to_bytes());
        final_out.extend_from_slice(tags::HEADER_DEBUG_SYMBOLS_START);
        for (label, addr) in self.linked_refs.iter() {
            final_out.extend_from_slice(tags::HEADER_DEBUG_SYMBOLS_ENTRY_ADDR);
            final_out.extend_from_slice(&addr.to_bytes());
            final_out.extend_from_slice(label.name.as_bytes());
            final_out.extend_from_slice(tags::HEADER_DEBUG_SYMBOLS_ENTRY_END);
        }
        final_out.extend_from_slice(tags::HEADER_DEBUG_SYMBOLS_END);
        final_out.extend_from_slice(tags::HEADER_END);

        final_out.extend_from_slice(&self.output);
        self.output = final_out;
        println!("Assembled program is {} bytes long.", self.output.len());

        Ok(self.output.clone())
    }
}
