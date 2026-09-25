#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
condact.py —— 嵌套块配置的条件激活求值工具（仅依赖 Python 标准库）

配置格式
--------
    # 注释以 # 开头
    var debug = true                 # 输入变量：true/false/数字/"字符串"
    var env   = "prod"

    block web @ (env == "prod" 且 max_conn >= 50) {   # @(...) 为激活条件，可省略（默认恒真）
        block tls @ (非 debug 或 @web) {              # @块名 引用另一块的激活状态
        }
    }

条件表达式（优先级：非 > 且 > 或，可用括号，也支持 not/and/or 与 !/&&/||）
    原子      := true | false | 变量 | 变量 (==|!=|>|<|>=|<=) 字面量 | @块名 | (条件)
    组合      := 非 条件 | 条件 且 条件 | 条件 或 条件        （且/或 短路求值）

激活语义
--------
    块激活 <=> 自身条件为真 且 所有祖先块均激活。
    条件中 @块名 会把被引用块的激活状态传播进来（按需递归求值）。

错误报告（均带行号）
--------------------
    * 条件引用未定义的变量 / 块          —— 报告行号，该块按未激活处理
    * 激活依赖成环（甲->乙->甲）          —— 报告完整循环链及每环行号，环上块全部不激活
    * 嵌套层级不闭合 / 多余的 "}"        —— 报告块起始行号

用法
----
    python3 condact.py 配置文件 [...]     # 求值指定配置文件（"-" 表示标准输入）
    python3 condact.py --demo             # 运行内置样例（正常 / 错误定位 / 未闭合）
"""

import ast
import re
import sys

# ---------------------------------------------------------------- 词法分析

TOKEN_RE = re.compile(r"""
      (?P<ws>[ \t\r\n]+)
    | (?P<comment>\#[^\n]*)
    | (?P<string>"(?:[^"\\\n]|\\.)*")
    | (?P<number>\d+(?:\.\d+)?)
    | (?P<op>==|!=|>=|<=|&&|\|\||[!@{}()=><])
    | (?P<ident>[A-Za-z_一-鿿][0-9A-Za-z_一-鿿]*)
    | (?P<bad>.)
""", re.VERBOSE)

KEYWORDS = {"var", "block", "true", "false",
            "且", "或", "非", "and", "or", "not"}


class ConfigError(Exception):
    """带行号的配置（解析）错误。"""

    def __init__(self, line, msg):
        self.line = line
        super().__init__("第 {} 行: {}".format(line, msg))


def tokenize(text):
    """把配置文本切成 (类别, 词素, 行号) 序列。"""
    tokens = []
    for m in TOKEN_RE.finditer(text):
        kind = m.lastgroup
        if kind in ("ws", "comment"):
            continue
        line = text.count("\n", 0, m.start()) + 1
        word = m.group()
        if kind == "bad":
            raise ConfigError(line, "无法识别的字符 {!r}".format(word))
        if kind == "ident" and word in KEYWORDS:
            kind = "kw"
        tokens.append((kind, word, line))
    tokens.append(("eof", "", text.count("\n") + 1))
    return tokens


# ---------------------------------------------------------------- 语法分析

class Block(object):
    __slots__ = ("name", "line", "cond", "parent", "children")

    def __init__(self, name, line, cond, parent):
        self.name = name
        self.line = line
        self.cond = cond          # 条件 AST，None 表示恒真
        self.parent = parent
        self.children = []


class Parser(object):
    """递归下降解析：program := stmt* ; stmt := var 定义 | block 定义。"""

    def __init__(self, tokens):
        self.toks = tokens
        self.pos = 0
        self.vars = {}      # 变量名 -> (值, 行号)
        self.blocks = {}    # 块名 -> Block（全局唯一命名）
        self.order = []     # 全部块，按文档顺序

    # ---- 词法单元工具 ----
    def peek(self):
        return self.toks[self.pos]

    def advance(self):
        tok = self.toks[self.pos]
        self.pos += 1
        return tok

    def at_kw(self, word):
        t = self.peek()
        return t[0] == "kw" and t[1] == word

    def at_op(self, sym):
        t = self.peek()
        return t[0] == "op" and t[1] == sym

    def expect_op(self, sym, msg):
        t = self.advance()
        if t[0] != "op" or t[1] != sym:
            raise ConfigError(t[2], msg)

    # ---- 语句 ----
    def parse(self):
        while self.peek()[0] != "eof":
            if self.at_op("}"):
                t = self.advance()
                raise ConfigError(t[2], '多余的 "}"（没有与之匹配的块）')
            self.parse_stmt(None)
        return self

    def parse_stmt(self, parent):
        if self.at_kw("var"):
            self.parse_var()
        elif self.at_kw("block"):
            self.parse_block(parent)
        else:
            t = self.peek()
            raise ConfigError(t[2], "期望 var 或 block 语句，得到 {!r}".format(t[1]))

    def parse_var(self):
        self.advance()  # var
        kind, name, line = self.advance()
        if kind != "ident":
            raise ConfigError(line, "var 后应为变量名")
        self.expect_op("=", '变量 "{}" 的定义缺少 "="'.format(name))
        value, _ = self.parse_literal()
        if name in self.vars:
            raise ConfigError(line, '变量 "{}" 重复定义（首次定义在第 {} 行）'
                              .format(name, self.vars[name][1]))
        self.vars[name] = (value, line)

    def parse_block(self, parent):
        self.advance()  # block
        kind, name, line = self.advance()
        if kind != "ident":
            raise ConfigError(line, "block 后应为块名")
        if name in self.blocks:
            raise ConfigError(line, '块 "{}" 重复定义（首次定义在第 {} 行）'
                              .format(name, self.blocks[name].line))
        cond = None
        if self.at_op("@"):
            self.advance()
            self.expect_op("(", '块 "{}" 的条件应写作 @ (条件表达式)'.format(name))
            cond = self.parse_cond()
            self.expect_op(")", '块 "{}" 的条件缺少 ")"'.format(name))
        self.expect_op("{", '块 "{}" 缺少 "{{"'.format(name))
        blk = Block(name, line, cond, parent)
        self.blocks[name] = blk
        self.order.append(blk)
        if parent is not None:
            parent.children.append(blk)
        while not self.at_op("}"):
            if self.peek()[0] == "eof":
                raise ConfigError(line, '块 "{}" 未闭合（缺少 "}}"）'.format(name))
            self.parse_stmt(blk)
        self.advance()  # }

    # ---- 条件表达式（优先级：非 > 且 > 或）----
    def parse_cond(self):
        return self.parse_or()

    def parse_or(self):
        node = self.parse_and()
        while self.at_kw("或") or self.at_kw("or") or self.at_op("||"):
            self.advance()
            node = ("or", node, self.parse_and())
        return node

    def parse_and(self):
        node = self.parse_not()
        while self.at_kw("且") or self.at_kw("and") or self.at_op("&&"):
            self.advance()
            node = ("and", node, self.parse_not())
        return node

    def parse_not(self):
        if self.at_kw("非") or self.at_kw("not") or self.at_op("!"):
            self.advance()
            return ("not", self.parse_not())
        return self.parse_atom()

    def parse_atom(self):
        kind, word, line = self.peek()
        if self.at_op("("):
            self.advance()
            node = self.parse_cond()
            self.expect_op(")", '条件表达式缺少 ")"')
            return node
        if self.at_op("@"):
            self.advance()
            k, name, at_line = self.advance()
            if k != "ident":
                raise ConfigError(at_line, '"@" 后应为块名')
            return ("blockref", name, line)
        if kind == "kw" and word in ("true", "false"):
            self.advance()
            return ("const", word == "true")
        if kind == "ident":
            self.advance()
            nk, nv, _ = self.peek()
            if nk == "op" and nv in ("==", "!=", ">", "<", ">=", "<="):
                self.advance()
                lit, _ = self.parse_literal()
                return ("cmp", word, nv, lit, line)
            return ("var", word, line)
        raise ConfigError(line, "条件表达式中出现意外的 {!r}".format(word))

    def parse_literal(self):
        kind, word, line = self.advance()
        if kind == "string":
            return ast.literal_eval(word), line
        if kind == "number":
            return (float(word) if "." in word else int(word)), line
        if kind == "kw" and word in ("true", "false"):
            return word == "true", line
        raise ConfigError(line, "期望字面量（字符串/数字/true/false），得到 {!r}".format(word))


# ---------------------------------------------------------------- 求值

class _Failed(Exception):
    """求值失败（错误已记录），仅作内部控制流使用。"""


def _compare(op, left, right):
    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    if op == ">":
        return left > right
    if op == "<":
        return left < right
    if op == ">=":
        return left >= right
    if op == "<=":
        return left <= right
    raise AssertionError(op)


class Evaluator(object):
    """按需（惰性）求值块激活状态；三色标记 DFS 检测依赖环。"""

    def __init__(self, variables, blocks):
        self.vars = variables      # 变量名 -> (值, 行号)
        self.blocks = blocks       # 块名 -> Block
        self.state = {}            # 块名 -> "visiting" / "done"
        self.active = {}           # 块名 -> bool
        self.stack = []            # 正在求值的块名栈（成环检测用）
        self.errors = []           # 求值期错误（字符串，含行号）

    def run(self, order):
        for blk in order:
            self.activate(blk)

    def activate(self, blk):
        """返回块的最终激活状态；任何失败都记为未激活且不向上抛。"""
        state = self.state.get(blk.name)
        if state == "done":
            return self.active[blk.name]
        if state == "visiting":
            idx = self.stack.index(blk.name)
            chain = self.stack[idx:] + [blk.name]
            desc = " -> ".join("{}(第{}行)".format(n, self.blocks[n].line)
                               for n in chain)
            self.errors.append("激活依赖成环: " + desc)
            raise _Failed
        self.state[blk.name] = "visiting"
        self.stack.append(blk.name)
        ok = False
        try:
            ok = True
            if blk.parent is not None:          # 祖先未激活则子孙必不激活
                ok = self.activate(blk.parent)
            if ok and blk.cond is not None:
                ok = bool(self.eval_cond(blk.cond))
        except _Failed:
            ok = False
        finally:
            self.stack.pop()
            self.state[blk.name] = "done"
            self.active[blk.name] = ok
        return ok

    def eval_cond(self, node):
        kind = node[0]
        if kind == "const":
            return node[1]
        if kind == "not":
            return not self.eval_cond(node[1])
        if kind == "and":                        # 短路求值
            return self.eval_cond(node[1]) and self.eval_cond(node[2])
        if kind == "or":
            return self.eval_cond(node[1]) or self.eval_cond(node[2])
        if kind == "var":
            name, line = node[1], node[2]
            if name not in self.vars:
                self.errors.append('第 {} 行: 条件引用了未定义的变量 "{}"'
                                   .format(line, name))
                raise _Failed
            return bool(self.vars[name][0])
        if kind == "blockref":
            name, line = node[1], node[2]
            blk = self.blocks.get(name)
            if blk is None:
                self.errors.append('第 {} 行: 条件引用了未定义的块 "@{}"'
                                   .format(line, name))
                raise _Failed
            return self.activate(blk)            # 激活状态沿引用传播
        if kind == "cmp":
            _, name, op, lit, line = node
            if name not in self.vars:
                self.errors.append('第 {} 行: 条件引用了未定义的变量 "{}"'
                                   .format(line, name))
                raise _Failed
            left = self.vars[name][0]
            try:
                return _compare(op, left, lit)
            except TypeError:
                self.errors.append("第 {} 行: 无法比较 {!r} {} {!r}（类型不匹配）"
                                   .format(line, left, op, lit))
                raise _Failed
        raise AssertionError(kind)


# ---------------------------------------------------------------- 报告

def block_path(blk):
    parts = []
    while blk is not None:
        parts.append(blk.name)
        blk = blk.parent
    return "/".join(reversed(parts))


def run_config(title, text, out=sys.stdout):
    out.write("===== {} =====\n".format(title))
    try:
        parser = Parser(tokenize(text)).parse()
    except ConfigError as exc:
        out.write("解析错误: {}\n\n".format(exc))
        return 1
    ev = Evaluator(parser.vars, parser.blocks)
    ev.run(parser.order)
    act = [b for b in parser.order if ev.active.get(b.name)]
    inact = [b for b in parser.order if not ev.active.get(b.name)]
    out.write("激活块（{}）:\n".format(len(act)))
    for b in act:
        out.write("  [激活]   {:<16} 第{}行\n".format(block_path(b), b.line))
    out.write("未激活块（{}）:\n".format(len(inact)))
    for b in inact:
        out.write("  [未激活] {:<16} 第{}行\n".format(block_path(b), b.line))
    if ev.errors:
        out.write("错误（{}）:\n".format(len(ev.errors)))
        for msg in ev.errors:
            out.write("  {}\n".format(msg))
    out.write("\n")
    return 1 if ev.errors else 0


# ---------------------------------------------------------------- 内置样例

SAMPLE_OK = """\
# ---- 输入变量 ----
var debug = true
var env = "prod"
var max_conn = 100

# web：生产环境且连接数达标时激活
block web @ (env == "prod" 且 max_conn >= 50) {
    # tls：非调试模式，或父块 web 已激活（引用父块激活状态）
    block tls @ (非 debug 或 @web) {
    }
}

# report：tls 激活且处于调试模式（激活状态跨块传播）
block report @ (@tls 且 debug) {
}

# cache：非 dev 环境即激活
block cache @ (env != "dev") {
}

# legacy：仅 staging 环境激活
block legacy @ (env == "staging") {
}
"""

SAMPLE_ERR_REFS = """\
var debug = true

block 甲 @ (@乙) {
}

block 乙 @ (@甲 且 debug) {
}

block 丙 @ (unknown_flag 且 debug) {
}

block 丁 @ (@ghost 或 debug) {
}
"""

SAMPLE_ERR_NEST = """\
var debug = true

block outer @ (debug) {
    block inner @ (true) {
    }
# 此处忘记用 } 闭合 outer
"""


def main(argv):
    if len(argv) < 2 or argv[1] == "--demo":
        rc = 0
        rc |= run_config("样例 1：正常配置（输出激活清单）", SAMPLE_OK)
        rc |= run_config("样例 2：未定义引用 + 依赖成环（错误定位）", SAMPLE_ERR_REFS)
        rc |= run_config("样例 3：嵌套层级不闭合", SAMPLE_ERR_NEST)
        return 0 if argv[1:] == ["--demo"] else rc
    rc = 0
    for path in argv[1:]:
        if path == "-":
            rc |= run_config("<stdin>", sys.stdin.read())
        else:
            with open(path, encoding="utf-8") as fh:
                rc |= run_config(path, fh.read())
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
