const std = @import("std");
const builtin = std.builtin;
const mem = std.mem;
const fmt = std.fmt;

extern fn boot_printi(a: u64) void;
extern fn boot_printc(a: u8) void;
extern fn boot_hlt() noreturn;
extern fn boot_und() noreturn;
extern fn boot_write_pt(pt: u64, fa_bump: u64) noreturn;

fn boot_println(s: []const u8) void {
    for (s) |c| {
        boot_printc(c);
    }
    boot_printc('\n');
}

const PAGE_SIZE: u64 = 4096;
const PHYS_OFFSET: u64 = 0xffff800000000000;
const KERNEL_OFFSET_PHYS: u64 = 0x200000;
const KERNEL_OFFSET_VIRT: u64 = KERNEL_OFFSET_PHYS | PHYS_OFFSET;
const KERNEL_END_PHYS: u64 = 0x500000;
const FRAME_ALLOC_START: u64 = KERNEL_END_PHYS + PAGE_SIZE;
const KERNEL_END_VIRT: u64 = KERNEL_END_PHYS | PHYS_OFFSET;
const ENTRIES_PER_TABLE: usize = 512; // each entry is a u64 (8 bytes)

const BootFrame = packed struct {
    const Self = @This();
    value: u64,

    pub fn init(value: u64) Self {
        return Self{ .value = value };
    }

    pub fn addr_value(self: Self) u64 {
        return self.value & 0x000f_ffff_ffff_f000;
    }
    pub fn pt_flags(self: Self) u64 {
        return self.value & 0x0fff;
    }
};

const BootVirtAddr = struct {
    const Self = @This();
    value: u64,

    pub fn init(value: u64) Self {
        return Self{ .value = value };
    }

    pub fn as_hhdm_phys(self: Self) u64 {
        return self.value - PHYS_OFFSET;
    }

    pub fn pt4_idx(self: Self) usize {
        return ((self.value / 4096) >> 27) & 0x1ff;
    }

    pub fn pt3_idx(self: Self) usize {
        return ((self.value / 4096) >> 18) & 0x1ff;
    }

    pub fn pt2_idx(self: Self) usize {
        return ((self.value / 4096) >> 9) & 0x1ff;
    }
    pub fn pt1_idx(self: Self) usize {
        return (self.value / 4096) & 0x1ff;
    }
    pub fn page_offset(self: Self) u64 {
        return self.value & 0xfff;
    }
};

pub fn BootPageTable(comptime Level: u8) type {
    return packed struct {
        const Self = @This();
        table: *[ENTRIES_PER_TABLE]BootFrame,

        pub noinline fn init(frame: BootFrame) Self {
            return Self{ .table = @intToPtr(*[ENTRIES_PER_TABLE]BootFrame, frame.addr_value()) };
        }

        pub noinline fn next_table(self: *const Self, index: usize) ?BootPageTable(Level - 1) {
            if (self.table[index].pt_flags() & 0b1 != 0) {
                return BootPageTable(Level - 1).init(self.table[index]);
            } else {
                return null;
            }
        }

        pub noinline fn next_table_create(self: *Self, index: usize, flags: u64) BootPageTable(Level - 1) {
            return self.next_table(index) orelse {
                const allocated = frame_alloc.alloc();
                const frame = BootFrame.init(allocated | flags | 0b1);
                self.table[index] = frame;
                return BootPageTable(Level - 1).init(frame);
            };
        }
    };
}

pub noinline fn boot_map_to(pt4: *BootPageTable(4), virt: BootVirtAddr, frame: BootFrame, flags: u64) void {
    var pt3 = pt4.next_table_create(virt.pt4_idx(), flags);
    var pt2 = pt3.next_table_create(virt.pt3_idx(), flags);
    var pt1 = pt2.next_table_create(virt.pt2_idx(), flags);
    pt1.table[virt.pt1_idx()] = frame;
    pt1.table[virt.pt1_idx()].value |= flags | 0b1;
}

const BootFrameAllocator = struct {
    const Self = @This();
    bump: u64,

    pub fn init() Self {
        return Self{ .bump = FRAME_ALLOC_START };
    }

    pub noinline fn alloc(self: *Self) u64 {
        const val = self.bump;
        self.bump += PAGE_SIZE;
        return val;
    }
};

var frame_alloc: BootFrameAllocator = BootFrameAllocator.init();

noinline fn setup_paging(pt4_frame: BootFrame) BootPageTable(4) {
    var pt = BootPageTable(4).init(pt4_frame);
    var ident_frame: u64 = 0x100000;
    while (ident_frame < KERNEL_OFFSET_PHYS) {
        boot_map_to(&pt, BootVirtAddr.init(ident_frame), BootFrame.init(ident_frame), 0b1);
        boot_map_to(&pt, BootVirtAddr.init(ident_frame | PHYS_OFFSET), BootFrame.init(ident_frame), 0b1);
        ident_frame += PAGE_SIZE;
    }
    var frame: u64 = KERNEL_OFFSET_PHYS;
    while (frame < frame_alloc.bump) {
        boot_map_to(&pt, BootVirtAddr.init(frame | PHYS_OFFSET), BootFrame.init(frame), 0b1);
        frame += PAGE_SIZE;
    }

    return pt;
}

pub export fn bootloader_main(mem_size: u64) noreturn {
    boot_println("Physical memory size is:");
    boot_printi(mem_size);
    frame_alloc = BootFrameAllocator.init();
    const pt4_frame = BootFrame.init(frame_alloc.alloc());
    boot_println("Creating page table at:");
    boot_printi(pt4_frame.addr_value());
    var page_table = setup_paging(pt4_frame);
    _ = page_table;
    boot_println("Enabling paging.");
    boot_write_pt(pt4_frame.addr_value(), frame_alloc.bump);
}

pub fn panic(msg: []const u8, stack_trace: ?*builtin.StackTrace) noreturn {
    _ = msg;
    _ = stack_trace;
    // var buf: [1024 * 10]u8 = undefined;
    // const formatted = fmt.bufPrint(buf[0..], "Panic! {s}", .{msg}) catch "Panic!!!";
    boot_println("Panic!!!");
    boot_und();
}
