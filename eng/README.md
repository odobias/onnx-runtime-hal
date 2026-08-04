# eng/

Build system and packaging **metadata** (not build outputs).

| Path | Role |
|------|------|
| `projects/` | MSBuild `.vcxproj` files |
| `msbuild/` | Shared props/targets (C++23, backends, HAL consumer props) |
| `packaging/` | Runner catalogs and schemas; assembled packages go to `dist/` |
