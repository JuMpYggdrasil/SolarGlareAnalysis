# SolarGlareAnalysis

Solar-glare and ocular-hazard analysis for a large-scale PV array at the
Map Ta Phut cargo port (Rayong, Thailand).

The pipeline computes full-year, per-minute sun/reflection geometry, detects
glare at receptor points with an SGHAT-style polygon detector, and classifies
ocular hazard (Green/Yellow/Red). It also reports IEC 62471 (BS EN 62471) risk
groups and an ICNIRP 2013 cross-check, using solar-weighted (ASTM E903-style)
glass reflectivity and Beer–Lambert atmospheric attenuation. Outputs include
per-case CSVs, figures, an interactive run timeline, and a self-contained HTML
report.

## Quick start

```powershell
.venv\Scripts\python.exe scripts\stages\run_all.py
```

See `AGENTS.md` for commands, configuration, and gotchas, and
`AGENTS_APPENDIX.md` for detailed results and provenance.

## License

MIT
