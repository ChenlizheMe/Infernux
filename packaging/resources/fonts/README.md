# Infernux Hub fonts

The Hub uses one typeface, Space Grotesk, in three static weights
(Regular 400, Medium 500, Bold 700). They are instances of the Latin subset
the website ships in `docs/assets/fonts/space-grotesk-latin.woff2`, converted
to TrueType because Qt does not load WOFF2. Chinese text falls back to the
operating system's UI face.

| File | Weight | License |
| --- | --- | --- |
| `SpaceGrotesk-Regular.ttf` | 400 | `Space-Grotesk-OFL.txt` (SIL OFL 1.1) |
| `SpaceGrotesk-Medium.ttf` | 500 | `Space-Grotesk-OFL.txt` (SIL OFL 1.1) |
| `SpaceGrotesk-Bold.ttf` | 700 | `Space-Grotesk-OFL.txt` (SIL OFL 1.1) |

Regenerate with fontTools `varLib.instancer.instantiateVariableFont(font, {"wght": N})`,
then set name IDs 1/2/16/17 to "Space Grotesk" and the weight name.
