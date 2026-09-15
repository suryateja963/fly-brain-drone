# Worlds

Build these in the Webots editor, in ascending difficulty. Start from the
bundled Mavic 2 Pro sample world and set its controller to `fly_brain`.

| File | Purpose | Phase |
|---|---|---|
| `01_empty.wbt` | Sanity; waypoint flight baseline | 1 |
| `02_corridor.wbt` | Wall following | 4 |
| `03_pillars.wbt` | Pillar forest — the money shot | 4, 8 |
| `04_target_behind.wbt` | Target behind obstacles, full task | 6 |
| `05_cluttered_room.wbt` | Stress test | 8 |

Give surfaces visible texture. Blank walls produce no optic flow at all, and
the EMD layer will have nothing to work with.
