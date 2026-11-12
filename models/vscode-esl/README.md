# ESL syntax highlighting for VSCode

Syntax highlighting for **ESL** (Epidemic Simulator Language), the DSL compiled to C++
by the ESC compiler in the ESCAPE framework.

This is a TextMate grammar, hand-ported from the tree-sitter queries in
[NSSAC/tree-sitter-esl](https://github.com/NSSAC/tree-sitter-esl) (branch `v1`):
`grammar.js` and `queries/highlights.scm`.

## Why TextMate and not tree-sitter

VSCode has no public tree-sitter API for extensions, so the grammar cannot be reused
directly. TextMate is VSCode's native highlighting engine — the same one used by every
built-in language. ESL's syntax is keyword-led and positional, so essentially every rule
in `highlights.scm` is expressible as a regex without a parse tree.

Because this extension contains no code (no `main`), VSCode treats it as a **UI
extension**: over Remote-SSH it installs and runs on your *local* machine, so nothing
needs to be installed on the cluster.

## Install

**From the packaged VSIX:**

```bash
code --install-extension vscode-esl-0.1.0.vsix
```

Or in VSCode: `Extensions` → `...` menu → `Install from VSIX...`.

Over Remote-SSH, install it on the **local** side ("Install Locally" if prompted).

**For development,** symlink into the extensions folder and reload:

```bash
ln -s "$PWD" ~/.vscode/extensions/vscode-esl
```

## Develop

Open this folder in VSCode and press `F5` to launch the Extension Development Host,
then open any `.esl` file.

To debug scope assignment, run `Developer: Inspect Editor Tokens and Scopes` from the
command palette and hover any token.

## Rebuild the VSIX

```bash
npx @vscode/vsce package
```

## What is highlighted

Ported from `queries/highlights.scm`:

| Construct | Scope |
| --- | --- |
| `enum` name, field/param/return types, `state type` | `entity.name.type` |
| enum constants, transition/transmission states | `constant.other` |
| `def` names, distribution names, transition `p`/`dwell` targets | `entity.name.function` |
| function calls, `len` | `support.function` |
| `node key`, `static`, `save`, `target/source node key` | `storage.modifier` |
| `CUR_TICK`, `NUM_NODES`, `ALL_NODES`, `ABSOLUTE`, `RELATIVE`, … | `constant.language` |
| `{% ... %}`, `{{ ... }}` templates | `keyword.control.directive` |
| keywords, comments, strings, numbers, operators | standard scopes |

### Context sensitivity

Several ESL words are keywords only in specific positions, and the grammar respects
this rather than matching them globally:

- `p`, `v`, `mean`, `std`, `min`, `max`, `low`, `high` are keywords **only inside
  `distribution` blocks**, so `max(a, b)` in an expression stays a function call.
- `p` and `dwell` are keywords **only inside `transition` blocks**.
- `node` and `edge` open blocks at the start of a line, but are **types** in positions
  like `(v: node)` and `[n: node -> bool]`.

### Known deviations from `highlights.scm`

- All function calls are scoped `support.function`, mirroring the upstream query
  (`function_call` → `@function.builtin`), which does not distinguish user-defined
  functions from builtins. To separate them, split the `function-call` rule in
  `syntaxes/esl.tmLanguage.json`.
- The `contagion` name is scoped `entity.name.type`. Upstream has no rule for it, so
  tree-sitter leaves it uncolored — this appears to be an oversight in `highlights.scm`.
- In `p = 1.0 - baseline_death_prob`, the identifier is not scoped as a function. This
  matches tree-sitter, where the value is a binary expression rather than a bare
  reference.
