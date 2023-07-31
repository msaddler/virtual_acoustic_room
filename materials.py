import dataclasses


@dataclasses.dataclass
class Material:
    name: str
    absorption_coefficients: list
    mat_type: str

map_int_to_material = {
    # WALLS
    1: 'Brick',
    2: 'Concrete, painted',
    3: 'Window Glass',
    4: 'Marble',
    5: 'Plaster on Concrete',
    6: 'Plywood',
    7: 'Concrete block, coarse',
    8: 'Heavyweight drapery',
    9: 'Fiberglass wall treatment, 1 in',
    10: 'Fiberglass wall treatment, 7 in',
    11: 'Wood panelling on glass fiber blanket',
    # FLOORS
    12: 'Wood parquet on concrete',
    13: 'Linoleum',
    14: 'Carpet on concrete',
    15: 'Carpet on foam rubber padding',
    # CEILINGS
    16: 'Plaster, gypsum, or lime on lath',
    17: 'Acoustic tiles, 0.625", 16" below ceiling',
    18: 'Acoustic tiles, 0.5", 16" below ceiling',
    19: 'Acoustic tiles, 0.5" cemented to ceiling',
    20: 'Highly absorptive panels, 1", 16" below ceiling',
    # OTHERS
    21: 'Upholstered seats',
    22: 'Audience in upholstered seats',
    23: 'Grass',
    24: 'Soil',
    25: 'Water surface',
    26: 'Anechoic',
    27: 'Uniform (0.6) absorbtion coefficient',
    28: 'Uniform (0.2) absorbtion coefficient',
    29: 'Uniform (0.8) absorbtion coefficient',
    30: 'Uniform (0.14) absorbtion coefficient',
    31: 'Artificial - absorbs more at high freqs',
    32: 'Artificial with absorption higher in middle ranges',
    33: 'Artificial - absorbs more at low freqs',
}


walls = [
        [0.03, 0.03, 0.03, 0.04, 0.05, 0.07],  # 1  : Brick
        [0.10, 0.05, 0.06, 0.07, 0.09, 0.08],  # 2  : Concrete, painted
        [0.35, 0.25, 0.18, 0.12, 0.07, 0.04],  # 3  : Window Glass
        [0.01, 0.01, 0.01, 0.01, 0.02, 0.02],  # 4  : Marble
        [0.12, 0.09, 0.07, 0.05, 0.05, 0.04],  # 5  : Plaster on Concrete
        [0.28, 0.22, 0.17, 0.09, 0.10, 0.11],  # 6  : Plywood
        [0.36, 0.44, 0.31, 0.29, 0.39, 0.25],  # 7  : Concrete block, coarse
        [0.14, 0.35, 0.55, 0.72, 0.70, 0.65],  # 8  : Heavyweight drapery
        [0.08, 0.32, 0.99, 0.76, 0.34, 0.12],  # 9  : Fiberglass wall treatment, 1 in
        [0.86, 0.99, 0.99, 0.99, 0.99, 0.99],  # 10 : Fiberglass wall treatment, 7 in
        [0.40, 0.90, 0.80, 0.50, 0.40, 0.30],  # 11 : Wood panelling on glass fiber blanket
]
floors = [
        [0.04, 0.04, 0.07, 0.06, 0.06, 0.07],  # 12 : Wood parquet on concrete
        [0.02, 0.03, 0.03, 0.03, 0.03, 0.02],  # 13 : Linoleum
        [0.02, 0.06, 0.14, 0.37, 0.60, 0.65],  # 14 : Carpet on concrete
        [0.08, 0.24, 0.57, 0.69, 0.71, 0.73],  # 15 : Carpet on foam rubber padding
]
ceilings = [
        [0.14, 0.10, 0.06, 0.05, 0.04, 0.03],  # 16 : Plaster, gypsum, or lime on lath
        [0.25, 0.28, 0.46, 0.71, 0.86, 0.93],  # 17 : Acoustic tiles, 0.625", 16" below ceiling
        [0.52, 0.37, 0.50, 0.69, 0.79, 0.78],  # 18 : Acoustic tiles, 0.5", 16" below ceiling
        [0.10, 0.22, 0.61, 0.66, 0.74, 0.72],  # 19 : Acoustic tiles, 0.5" cemented to ceiling
        [0.58, 0.88, 0.75, 0.99, 1.00, 0.96],  # 20 : Highly absorptive panels, 1", 16" below ceiling
]
others = [
        [0.19, 0.37, 0.56, 0.67, 0.61, 0.59],  # 21 : Upholstered seats
        [0.39, 0.57, 0.80, 0.94, 0.92, 0.87],  # 22 : Audience in upholstered seats
        [0.11, 0.26, 0.60, 0.69, 0.92, 0.99],  # 23 : Grass
        [0.15, 0.25, 0.40, 0.55, 0.60, 0.60],  # 24 : Soil
        [0.01, 0.01, 0.01, 0.02, 0.02, 0.03],  # 25 : Water surface
        [1.00, 1.00, 1.00, 1.00, 1.00, 1.00],  # 26 : Anechoic
        # 27 : Uniform (0.6) absorbtion coefficient
        [0.60, 0.60, 0.60, 0.60, 0.60, 0.60],
        # 28 : Uniform (0.2) absorbtion coefficient
        [0.20, 0.20, 0.20, 0.20, 0.20, 0.20],
        # 29 : Uniform (0.8) absorbtion coefficient
        [0.80, 0.80, 0.80, 0.80, 0.80, 0.80],
        # 30 : Uniform (0.14) absorbtion coefficient
        [0.14, 0.14, 0.14, 0.14, 0.14, 0.14],
        # 31 : Artificial - absorbs more at high freqs
        [0.08, 0.08, 0.10, 0.10, 0.12, 0.12],
        # 32 : Artificial with absorption higher in middle ranges
        [0.05, 0.05, 0.20, 0.20, 0.10, 0.10],
        # 33 : Artificial  - absorbs more at low freqs
        [0.12, 0.12, 0.10, 0.10, 0.08, 0.08],
]



# from https://www.acoustic.ua/st/web_absorption_data_eng.pdf

masonry_walls = [
    Material("Rough concrete", [0.02, 0.03,0.03,0.03,0.04,0.07], "masonry wall"),
    Material("Smooth unpainted concrete", [0.01,0.01,0.02,0.02,0.02,0.05], "masonry wall"),
    Material("Smooth concrete, painted or glazed", [0.01,0.01,0.01,0.02,0.02,0.02], "masonry wall"),
    Material("Porous concrete blocks (no surface finish)", [0.05,0.05,0.05,0.08,0.14,0.2], "masonry wall"),
    Material("Clinker concrete (no surface finish)", [0.10,0.20,0.40,0.60,0.50,0.60], "masonry wall"),
    Material("Smooth brickwork with flush pointing", [0.02,0.03,0.03,0.04,0.05,0.07], "masonry wall"),
    Material("Smooth brickwork with flush pointing, painted", [0.01, 0.01, 0.02, 0.02, 0.02, 0.02], "masonry wall"),
    Material("Standard brickwork", [0.05, 0.04, 0.02, 0.04, 0.05, 0.05], "masonry wall"),
    Material("Brickwork, 10mm flush pointing", [0.08, 0.09, 0.12, 0.16, 0.22, 0.24], "masonry wall"),
    Material("Lime cement plaster on masonry wall", [0.08, 0.09, 0.12, 0.16, 0.22, 0.24], "masonry wall"),
    Material("Glaze plaster on masonry wall", [0.01, 0.01, 0.01, 0.02, 0.02, 0.02], "masonry wall"),
    Material("Painted plaster surface on masonry wall", [0.02, 0.02, 0.02, 0.02, 0.02, 0.02], "masonry wall"),
    Material("Plaster on masonry wall with wall paper on backing paper", [0.02, 0.03, 0.04, 0.05, 0.07, 0.08], "masonry wall"),
    Material("Ceramic tiles with smooth surface", [0.01, 0.01, 0.01, 0.02, 0.02, 0.02], "masonry wall"),
    Material("Breeze block", [0.20, 0.45, 0.60, 0.40, 0.45, 0.40], "masonry wall"),
    Material("Plaster on solid wall", [0.04, 0.05, 0.06, 0.08, 0.04, 0.06], "masonry wall"),
    Material("Plaster, lime or gypsum on solid backing", [0.03, 0.03, 0.02, 0.03, 0.04, 0.05], "masonry wall"),
]

studwork_lightweight_walls = [
    Material("Plasterboard on battens, 18mm airspace with glass wool", [0.30, 0.20, 0.15, 0.05, 0.05, 0.05], "studwork lightweight wall"),
    Material("Plasterboard on frame, 100mm airspace", [0.30, 0.12, 0.08, 0.06, 0.06, 0.05], "studwork lightweight wall"),
    Material("Plasterboard on frame, 100mm airspace with glass wool", [0.08, 0.11, 0.05, 0.03, 0.02, 0.03], "studwork lightweight wall"),
    Material("Plasterboard on 50mm battens", [0.29, 0.10, 0.05, 0.04, 0.07, 0.09], "studwork_lightweight_wall"),
    Material("Plasterboard on 25mm battens", [0.31, 0.33, 0.14, 0.10, 0.10, 0.12], "studwork_lightweight_wall"),
    Material("2 x plasterboard on frame, 50mm airspace with mineral wool", [0.15, 0.10, 0.06, 0.04, 0.04, 0.05], "studwork_lightweight_wall"),
    Material("Plasterboard on cellular core partition", [0.15, 0.00, 0.07, 0.00, 0.04, 0.05], "studwork_lightweight_wall"),
    Material("Plasterboard on frame 100mm cavity", [0.08, 0.11, 0.05, 0.03, 0.02, 0.03], "studwork_lightweight_wall"),
    Material("Plasterboard on frame, 100mm cavity with mineral wool", [0.30, 0.12, 0.08, 0.06, 0.06, 0.05], "studwork_lightweight_wall"),
    Material("2 x 13mm plasterboard on steel frame, 50mm mineral wool in cavity, surface painted", [0.15, 0.01, 0.06, 0.04, 0.04, 0.05])
]

glass_and_glazing = [
    Material("4mm glass", [0.30, 0.20, 0.10, 0.07, 0.05, 0.02], "glass and glazing"),
    Material("6mm glass", [0.10, 0.06, 0.04, 0.03, 0.02, 0.02], "glass and glazing"),
    Material("Double glazing, 2-3mm glass, 10mm air gap", [0.15, 0.05, 0.03, 0.03, 0.02, 0.02], "glass and glazing")
]

wood_and_wood_paneling = [
    Material("3-4mm plywood, 75mm cavity containing mineral wool", [0.50, 0.30, 0.10, 0.05, 0.05, 0.05], "wood and wood paneling"),
    Material("5mm plywood on battens, 50mm airspace filled", [0.40, 0.35, 0.20, 0.15, 0.05, 0.05], "wood and wood paneling"),
    Material("12mm plywood over 50mm airgap", [0.25, 0.05, 0.04, 0.03, 0.03, 0.02], "wood and wood paneling"),
    Material("12mm plywood over 150mm airgap", [0.28, 0.08, 0.07, 0.07, 0.09, 0.09], "wood and wood paneling"),
    Material("12mm plywood over 200mm airgap containing 50mm mineral wool", [0.14, 0.10, 0.10, 0.08, 0.10, 0.08], "wood and wood paneling"),
    Material("Plywood mounted solidly", [0.05, None, 0.05, None, 0.05, 0.05], "wood and wood paneling"), # Will maybe leave this out since no numbers @ 250 and 1000hz
    Material("12mm plywood in framework with 30mm airspace behind", [0.35, 0.20, 0.15, 0.10, 0.05, 0.05], "wood and wood paneling"),
    Material("12mm plywood in framework with 30mm airspace containing glass wool", [0.40, 0.20, 0.15, 0.10, 0.10, 0.05], "wood and wood paneling"),
    Material("Plywood, hardwood panels over 25mm airspace on solid backing", [0.30, 0.20, 0.15, 0.10, 0.10, 0.05], "wood and wood paneling"),
    Material("Plywood, hardwood panels over 25mm airspace on solid backing with absorbent material in air space", [0.40, 0.25, 0.15, 0.10, 0.10, 0.05], "wood and wood paneling"),
    Material("12mm wood panelling on 25mm battens", [0.31, 0.33, 0.14, 0.10, 0.10, 0.12], "wood and wood paneling"),
    Material("Timber boards, 100mm wide, 10mm gaps, 500mm airspace with mineral wool", [0.05, 0.25, 0.60, 0.15, 0.05, 0.10], "wood and wood paneling"),
    Material("t & g board on frame, 50mm airspace with mineral wool", [0.25, 0.15, 0.10, 0.09, 0.08, 0.07], "wood and wood paneling"),
    Material("16-22mm t&g wood on 50mm cavity filled with mineral wool", [0.25, 0.15, 0.10, 0.09, 0.08, 0.07], "wood and wood paneling"),
    Material("Cedar, slotted and profiled on battens mineral wool in airspace", [0.20, 0.62, 0.98, 0.62, 0.21, 0.15], "wood and wood paneling"),
    Material("Wood boards on on joists or battens", [0.15, 0.20, 0.10, 0.10, 0.10, 0.10], "wood and wood paneling"),
    Material("20mm dense veneered chipboard over 100mm airgap", [0.03, 0.05, 0.04, 0.03, 0.03, 0.02]. "wood and wood paneling"),
    Material("20mm dense veneered chipboard over 200mm airgap", [0.06, 0.10, 0.08, 0.09, 0.07, 0.04], "wood and wood paneling"),
    Material("20mm dense veneered chipboard over 250mm airgap containing 50mm mineral wool", [0.12, 0.10, 0.08, 0.07, 0.10, 0.08], "wood and wood paneling"),
    Material("6mm wood fibre board, cavity > 100mm, empty", [0.30, 0.20, 0.20, 0.10, 0.05, 0.05], "wood and wood paneling"),
    Material("22mm chipboard, 50mm cavity filled with mineral wool", [0.12, 0.04, 0.06, 0.05, 0.05, 0.05], "wood and wood paneling"),
    Material("Acoustic timber wall panelling", [0.18, 0.34, 0.42, 0.59, 0.83, 0.68], "wood and wood paneling"),
    Material("Hardwood, mahogany", [0.19, 0.23, 0.25, 0.30, 0.37, 0.42], "wood and wood paneling"),
    Material("Chipboard on 16mm battens", [0.20, 0.25, 0.20, 0.20, 0.15, 0.20], "wood and wood paneling"),
    Material("Chipboard on frame, 50mm airspace with mineral wool", [0.12, 0.04, 0.06, 0.05, 0.05, 0.05], "wood and wood paneling"),

]

