#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
condblock.py — 条件块配置求值工具(纯 Python 标准库, 单文件)

用法:
    python3 condblock.py 配置文件      # 从文件读取
    python3 condblock.py < 配置文件    # 从标准输入读取

配置语法:
    # 注释
    let debug = true                    # 输入变量(布尔 / 数字 / "字符串"), 只允许顶层
    block web when <条件> {             # 命名块, 可任意嵌套; when 可省略(视为 true)
        block api when <条件> {
        }
    }

条件表达式(优先级: 括号 > 非 > 比较 > 且 > 或):
    变量名                              引用 let 定义的变量
    block 块名                          引用另一个块的"激活状态"(布尔)
    非 A                                逻辑非
    A 且 B / A 或 B                     逻辑与 / 逻辑或
    a == b  !=  <  <=  >  >=            比较运算(数字与数字、字符串与字符串)
    true / false / 123 / 1.5 / "文本"   字面量
    ( ... )                             括号

语义约定:
    1. 块激活 = 自身条件为真 且 所有祖先块均激活(嵌套块依附于父块)。
    2. 条件中引用其他块时, 按需递归求值该块, 激活状态沿依赖边传播。
    3. 依赖成环时报告完整循环链, 环上的块按未激活处理, 继续评估其余块。
    4. 引用未定义的变量/块、层级不闭合等都会带行号报告; 有错误时退出码为 1。
"""

import ast
import re
import sys
from dataclasses import dataclass, field


# ================================================================ 词法分析

TOKEN_RE = re.compile(r"""
      (?P<space>\s+)
    | (?P<comment>\#.*)
    | (?P<number>\d+(?:\.\d+)?)
    | (?P<string>"(?:[^"\\\n]|\\.)*")
    | (?P<op>==|!=|<=|>=|<|>|=)
    | (?P<punct>[{}()])
    | (?P<word>[^\s{}()#=<>!"]+)
    | (?P<bad>.)
""", re.VERBOSE)

LOGIC_WORDS = {"且": "AND", "或": "OR", "非": "NOT"}
KEYWORDS = {"let": "LET", "block": "BLOCK", "when": "WHEN",
            "true": "TRUE", "false": "FALSE"}


@dataclass
class Token:
    kind: str
    text: str
    line: int


def tokenize(text):
    """把源文本切成 Token 流; 每个 Token 都带行号, 供后续精确定位错误。"""
    tokens, errors = [], []
    for m in TOKEN_RE.finditer(text):
        kind, value = m.lastgroup, m.group()
        line = text.count("\n", 0, m.start()) + 1
        if kind in ("space", "comment"):
            continue
        if kind == "bad":
            errors.append((line, "无法识别的字符 %r" % value))
            continue
        if kind in ("op", "punct"):
            kind = value                      # 符号直接用自身当 kind
        elif kind in ("number", "string"):
            kind = kind.upper()
        elif kind == "word":
            kind = LOGIC_WORDS.get(value) or KEYWORDS.get(value) or "IDENT"
        tokens.append(Token(kind, value, line))
    tokens.append(Token("EOF", "", text.count("\n") + 1))
    return tokens, errors


# ================================================================ 语法树

@dataclass
class Literal:
    value: object
    line: int

@dataclass
class VarRef:            # 引用输入变量
    name: str
    line: int

@dataclass
class BlockRef:          # 引用另一个块的激活状态
    name: str
    line: int

@dataclass
class Not:
    operand: object
    line: int

@dataclass
class BinOp:             # 且 / 或
    op: str
    left: object
    right: object
    line: int

@dataclass
class Compare:
    op: str
    left: object
    right: object
    line: int

@dataclass
class Block:
    name: str
    cond: object                       # 条件表达式 AST, 可为 None
    line: int
    parent: object = None
    children: list = field(default_factory=list)

@dataclass
class Program:
    variables: dict                    # name -> (value, line)
    roots: list
    errors: list                       # [(line, msg), ...]


class ConfigError(Exception):
    """致命语法错误: 携带行号, 抛出即终止解析。"""
    def __init__(self, line, msg):
        super().__init__(msg)
        self.line, self.msg = line, msg


class Parser:
    def __init__(self, tokens, errors):
        self.tokens, self.pos = tokens, 0
        self.errors = errors
        self.variables, self.roots = {}, []

    def peek(self):
        return self.tokens[self.pos]

    def advance(self):
        tok = self.tokens[self.pos]
        if tok.kind != "EOF":
            self.pos += 1
        return tok

    def expect(self, kind, what):
        tok = self.peek()
        if tok.kind != kind:
            raise ConfigError(tok.line, "期望%s, 却遇到 %r" % (what, tok.text))
        return self.advance()

    # ---- 顶层 ----
    def parse(self):
        while self.peek().kind != "EOF":
            tok = self.peek()
            if tok.kind == "LET":
                self.parse_let()
            elif tok.kind == "BLOCK":
                self.roots.append(self.parse_block(None))
            elif tok.kind == "}":
                self.errors.append((tok.line, "多余的 '}', 没有匹配的 '{'"))
                self.advance()
            else:
                raise ConfigError(tok.line, "无法解析 %r" % tok.text)
        return Program(self.variables, self.roots, self.errors)

    def parse_let(self):
        self.advance()                                   # let
        name = self.expect("IDENT", "变量名")
        self.expect("=", "'='")
        value = self.parse_literal()
        if name.text in self.variables:
            first = self.variables[name.text][1]
            self.errors.append((name.line,
                "变量 '%s' 重复定义(首次定义在第%d行)" % (name.text, first)))
        else:
            self.variables[name.text] = (value, name.line)

    def parse_literal(self):
        tok = self.advance()
        if tok.kind == "NUMBER":
            return float(tok.text) if "." in tok.text else int(tok.text)
        if tok.kind == "STRING":
            try:
                return ast.literal_eval(tok.text)
            except (ValueError, SyntaxError):
                raise ConfigError(tok.line, "非法的字符串字面量")
        if tok.kind == "TRUE":
            return True
        if tok.kind == "FALSE":
            return False
        raise ConfigError(tok.line, "期望字面量, 却遇到 %r" % tok.text)

    # ---- 块(可嵌套) ----
    def parse_block(self, parent):
        start = self.advance()                           # block
        name = self.expect("IDENT", "块名")
        cond = None
        if self.peek().kind == "WHEN":
            self.advance()
            cond = self.parse_expr()
        blk = Block(name.text, cond, start.line, parent)
        self.expect("{", "'{'")
        while True:
            tok = self.peek()
            if tok.kind == "}":
                self.advance()
                return blk
            if tok.kind == "EOF":                        # 层级不闭合
                self.errors.append((start.line,
                    "块 '%s' 的 '{' 到文件末尾仍未闭合" % name.text))
                return blk
            if tok.kind == "BLOCK":
                blk.children.append(self.parse_block(blk))
            elif tok.kind == "LET":
                self.errors.append((tok.line, "'let' 只能出现在顶层, 已忽略"))
                self.advance()
            else:
                raise ConfigError(tok.line, "块内出现意外的 %r" % tok.text)

    # ---- 条件表达式: 或 < 且 < 非 < 比较 < 原子 ----
    def parse_expr(self):
        return self.parse_or()

    def parse_or(self):
        node = self.parse_and()
        while self.peek().kind == "OR":
            tok = self.advance()
            node = BinOp("或", node, self.parse_and(), tok.line)
        return node

    def parse_and(self):
        node = self.parse_not()
        while self.peek().kind == "AND":
            tok = self.advance()
            node = BinOp("且", node, self.parse_not(), tok.line)
        return node

    def parse_not(self):
        tok = self.peek()
        if tok.kind == "NOT":
            self.advance()
            return Not(self.parse_not(), tok.line)
        return self.parse_compare()

    def parse_compare(self):
        node = self.parse_primary()
        tok = self.peek()
        if tok.kind in ("==", "!=", "<", "<=", ">", ">="):
            self.advance()
            node = Compare(tok.kind, node, self.parse_primary(), tok.line)
        return node

    def parse_primary(self):
        tok = self.peek()
        if tok.kind == "IDENT":
            self.advance()
            return VarRef(tok.text, tok.line)
        if tok.kind == "BLOCK":                          # block 块名 -> 块引用
            self.advance()
            name = self.expect("IDENT", "块名")
            return BlockRef(name.text, tok.line)
        if tok.kind in ("NUMBER", "STRING", "TRUE", "FALSE"):
            return Literal(self.parse_literal(), tok.line)
        if tok.kind == "(":
            self.advance()
            node = self.parse_expr()
            self.expect(")", "')'")
            return node
        raise ConfigError(tok.line, "条件表达式中出现意外的 %r" % tok.text)


# ================================================================ 求值

class Evaluator:
    """懒求值 + 记忆化 + DFS 在栈检测环。"""

    def __init__(self, program):
        self.variables = program.variables
        self.errors = program.errors
        self.blocks = {}            # name -> Block
        self.order = []             # 源文件出现顺序
        self.state = {}             # name -> bool (记忆化结果)
        self.stack = []             # 正在求值的块名栈(用于环检测)
        self.reported_cycles = set()
        self.roots = program.roots
        for root in program.roots:
            self._register(root)

    def _register(self, blk):
        if blk.name in self.blocks:
            first = self.blocks[blk.name].line
            self.errors.append((blk.line,
                "块 '%s' 重复定义(首次定义在第%d行)" % (blk.name, first)))
        else:
            self.blocks[blk.name] = blk
            self.order.append(blk.name)
        for child in blk.children:
            self._register(child)

    def error(self, line, msg):
        self.errors.append((line, msg))

    # ---- 块激活状态: 依赖传播的入口 ----
    def is_active(self, name, ref_line=None):
        if name in self.state:                     # 记忆化命中
            return self.state[name]
        blk = self.blocks.get(name)
        if blk is None:
            return False                           # 未定义块已在 BlockRef 处报错
        if name in self.stack:                     # 回到栈上的块 -> 环
            i = self.stack.index(name)
            chain = self.stack[i:] + [name]
            key = frozenset(chain)
            if key not in self.reported_cycles:    # 同一个环只报一次
                self.reported_cycles.add(key)
                self.error(ref_line or blk.line,
                           "检测到循环依赖: " + " -> ".join(chain))
            return False                           # 环上的块按未激活处理
        self.stack.append(name)
        try:
            ok = True
            if blk.parent is not None:             # 祖先不激活则自身不激活
                ok = self.is_active(blk.parent.name, blk.line)
            if ok and blk.cond is not None:
                ok = self.truthy(self.eval_expr(blk.cond), blk.line)
            self.state[name] = ok
        finally:
            self.stack.pop()
        return self.state[name]

    # ---- 条件表达式求值 ----
    def eval_expr(self, node):
        if isinstance(node, Literal):
            return node.value
        if isinstance(node, VarRef):
            if node.name not in self.variables:
                self.error(node.line, "未定义的变量 '%s'" % node.name)
                return False
            return self.variables[node.name][0]
        if isinstance(node, BlockRef):
            if node.name not in self.blocks:
                self.error(node.line, "未定义的块 '%s'" % node.name)
                return False
            return self.is_active(node.name, node.line)   # 激活状态传播
        if isinstance(node, Not):
            return not self.truthy(self.eval_expr(node.operand), node.line)
        if isinstance(node, BinOp):
            # 故意不短路: 两侧都求值, 以便一次报告所有未定义引用
            left = self.truthy(self.eval_expr(node.left), node.line)
            right = self.truthy(self.eval_expr(node.right), node.line)
            return (left and right) if node.op == "且" else (left or right)
        if isinstance(node, Compare):
            return self.compare(node, self.eval_expr(node.left),
                                self.eval_expr(node.right))
        raise AssertionError("未知表达式节点: %r" % (node,))

    def truthy(self, value, line):
        if isinstance(value, bool):
            return value
        self.error(line, "期望布尔值, 实际得到 %r" % (value,))
        return False

    def compare(self, node, left, right):
        if node.op == "==":
            return left == right
        if node.op == "!=":
            return left != right
        number = (int, float)
        both_num = (not isinstance(left, bool) and not isinstance(right, bool)
                    and isinstance(left, number) and isinstance(right, number))
        if not (both_num or type(left) is type(right)):
            self.error(node.line, "无法比较 %r 与 %r: 类型不匹配" % (left, right))
            return False
        return {"<": left < right, "<=": left <= right,
                ">": left > right, ">=": left >= right}[node.op]

    def evaluate_all(self):
        for name in self.order:                    # 按源文件顺序求值, 输出稳定
            self.is_active(name)


# ================================================================ 输出与入口

def print_active_tree(blocks, ev, depth=0, out=None):
    for blk in blocks:
        if ev.state.get(blk.name):
            out.append("  " * depth + blk.name)
            print_active_tree(blk.children, ev, depth + 1, out)


def run(text):
    tokens, errors = tokenize(text)
    try:
        program = Parser(tokens, errors).parse()
    except ConfigError as exc:
        errors.append((exc.line, exc.msg))
        return None, errors
    ev = Evaluator(program)
    ev.evaluate_all()
    return ev, errors


def main(argv):
    if len(argv) > 2 or (len(argv) == 2 and argv[1] in ("-h", "--help")):
        print(__doc__.strip())
        return 0 if len(argv) == 2 else 2
    if len(argv) == 2:
        try:
            with open(argv[1], encoding="utf-8") as f:
                text = f.read()
        except OSError as exc:
            print("无法读取文件: %s" % exc, file=sys.stderr)
            return 2
    else:
        text = sys.stdin.read()

    ev, errors = run(text)

    if ev is not None:
        lines = []
        print_active_tree(ev.roots, ev, 0, lines)
        active = [n for n in ev.order if ev.state.get(n)]
        print("== 激活块清单 (%d) ==" % len(active))
        print("\n".join(lines) if lines else "(无)")
    else:
        print("== 解析失败, 未进行求值 ==")

    if errors:
        print("\n== 错误 (%d) ==" % len(errors))
        for line, msg in sorted(errors):
            print("第%d行: %s" % (line, msg))
    else:
        print("\n无错误。")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
