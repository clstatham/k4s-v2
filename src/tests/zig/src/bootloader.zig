const std = @import("std");
const builtin = std.builtin;
const mem = std.mem;
const fmt = std.fmt;

extern fn printi(a: u64) void;
extern fn printc(a: u8) void;
extern fn hlt() noreturn;
extern fn und() noreturn;
extern fn write_pt(pt: u64) noreturn;

fn println(s: []const u8) void {
    for (s) |c| {
        printc(c);
    }
    printc('\n');
}

const FRAME_ALLOC_START: u64 = 0x1000;
const PHYS_OFFSET: u64 = 0x2000000;
const KERNEL_OFFSET_PHYS: u64 = 0x200000;
const KERNEL_OFFSET_VIRT: u64 = KERNEL_OFFSET_PHYS | PHYS_OFFSET;
const KERNEL_END_PHYS: u64 = 0x400000;
const KERNEL_END_VIRT: u64 = KERNEL_END_PHYS | PHYS_OFFSET;
const PAGE_SIZE: u64 = 4096;
const ENTRIES_PER_TABLE: usize = 512; // each entry is a u64 (8 bytes)

const Frame = packed struct {
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

const VirtAddr = struct {
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

pub fn PageTable(comptime Level: u8) type {
    return packed struct {
        const Self = @This();
        table: *[ENTRIES_PER_TABLE]Frame,

        pub noinline fn init(frame: Frame) Self {
            return Self{ .table = @intToPtr(*[ENTRIES_PER_TABLE]Frame, frame.addr_value()) };
        }

        pub noinline fn next_table(self: *const Self, index: usize) ?PageTable(Level - 1) {
            if (self.table[index].pt_flags() & 0b1 != 0) {
                return PageTable(Level - 1).init(self.table[index]);
            } else {
                return null;
            }
        }

        pub noinline fn next_table_create(self: *Self, index: usize, flags: u64, frame_alloc: *FrameAllocator) PageTable(Level - 1) {
            return self.next_table(index) orelse {
                const allocated = frame_alloc.alloc();
                const frame = Frame.init(allocated | flags | 0b1);
                self.table[index] = frame;
                return PageTable(Level - 1).init(frame);
            };
        }
    };
}

pub noinline fn translate(pt4: *PageTable(4), virt: VirtAddr) ?u64 {
    const pt3 = pt4.next_table(virt.pt4_idx()) orelse return null;
    const pt2 = pt3.next_table(virt.pt3_idx()) orelse return null;
    const pt1 = pt2.next_table(virt.pt2_idx()) orelse return null;
    const frame = pt1.table[virt.pt1_idx()];
    if (frame.pt_flags() & 0b1 == 0) {
        return null;
    }
    return frame.addr_value() + virt.page_offset();
}

pub noinline fn map_to(pt4: *PageTable(4), virt: VirtAddr, frame: Frame, flags: u64, frame_alloc: *FrameAllocator) void {
    var pt3 = pt4.next_table_create(virt.pt4_idx(), flags, frame_alloc);
    var pt2 = pt3.next_table_create(virt.pt3_idx(), flags, frame_alloc);
    var pt1 = pt2.next_table_create(virt.pt2_idx(), flags, frame_alloc);
    pt1.table[virt.pt1_idx()] = frame;
    pt1.table[virt.pt1_idx()].value |= flags | 0b1;
}

const FrameAllocator = struct {
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

noinline fn setup_paging(pt4_frame: Frame, frame_alloc: *FrameAllocator) PageTable(4) {
    var pt = PageTable(4).init(pt4_frame);
    var frame: u64 = 0x1000;
    while (frame < KERNEL_END_PHYS) {
        map_to(&pt, VirtAddr.init(frame | PHYS_OFFSET), Frame.init(frame), 0b1, frame_alloc);
        frame += PAGE_SIZE;
    }

    return pt;
}

pub export fn bootloader_main(mem_size: u64) noreturn {
    println("Physical memory size is:");
    printi(mem_size);
    println("Creating page table at:");
    var frame_alloc = FrameAllocator.init();
    const pt4_frame = Frame.init(frame_alloc.alloc());
    printi(pt4_frame.addr_value());

    var page_table = setup_paging(pt4_frame, &frame_alloc);
    _ = page_table;
    println("Enabling paging.");
    write_pt(@truncate(u64, pt4_frame.addr_value()));
    // println("Paging is enabled!");
    // map_to(&page_table, VirtAddr.init(0xbaadf00d), Frame.init(0xcafeb000), 0b1, &frame_alloc);
    // println("0xbaadf00d =>");
    // const trans = translate(&page_table, VirtAddr.init(0xbaadf00d));
    // if (trans == null) {
    //     println("Error translating 0xbaadf00d");
    // } else {
    //     printi(@intCast(u64, trans.?));
    // }
    // hlt();
}

pub export fn kernel_main() noreturn {
    printi(42069);
    hlt();
}

pub fn panic(msg: []const u8, stack_trace: ?*builtin.StackTrace) noreturn {
    _ = msg;
    _ = stack_trace;
    // var buf: [1024 * 10]u8 = undefined;
    // const formatted = fmt.bufPrint(buf[0..], "Panic! {s}", .{msg}) catch "Panic!!!";
    println("Panic!!!");
    und();
}
