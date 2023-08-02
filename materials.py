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
    Material("20mm dense veneered chipboard over 100mm airgap", [0.03, 0.05, 0.04, 0.03, 0.03, 0.02],"wood and wood paneling"),
    Material("20mm dense veneered chipboard over 200mm airgap", [0.06, 0.10, 0.08, 0.09, 0.07, 0.04], "wood and wood paneling"),
    Material("20mm dense veneered chipboard over 250mm airgap containing 50mm mineral wool", [0.12, 0.10, 0.08, 0.07, 0.10, 0.08], "wood and wood paneling"),
    Material("6mm wood fibre board, cavity > 100mm, empty", [0.30, 0.20, 0.20, 0.10, 0.05, 0.05], "wood and wood paneling"),
    Material("22mm chipboard, 50mm cavity filled with mineral wool", [0.12, 0.04, 0.06, 0.05, 0.05, 0.05], "wood and wood paneling"),
    Material("Acoustic timber wall panelling", [0.18, 0.34, 0.42, 0.59, 0.83, 0.68], "wood and wood paneling"),
    Material("Hardwood, mahogany", [0.19, 0.23, 0.25, 0.30, 0.37, 0.42], "wood and wood paneling"),
    Material("Chipboard on 16mm battens", [0.20, 0.25, 0.20, 0.20, 0.15, 0.20], "wood and wood paneling"),
    Material("Chipboard on frame, 50mm airspace with mineral wool", [0.12, 0.04, 0.06, 0.05, 0.05, 0.05], "wood and wood paneling"),

]

mineral_wool_and_foams = [
    Material("Melamine based foam 25mm", [0.09, 0.22, 0.54, 0.76, 0.88, 0.93], "mineral wool and foams"),
    Material("Melamine based foam 50mm", [0.18, 0.56, 0.96, 1.00, 1.00, 1.00], "mineral wool and foams"),
    Material("Glass wool 25mm 16 kg/m3", [0.12, 0.28, 0.55, 0.71, 0.74, 0.83], "mineral wool and foams"),
    Material("Glass wool 50mm, 16 kg/m3", [0.17, 0.45, 0.80, 0.89, 0.97, 0.94], "mineral wool and foams"),
    Material("Glass wool 75mm, 16 kg/m3", [0.30, 0.69, 0.94, 1.00, 1.00, 1.00], "mineral wool and foams"),
    Material("Glass wool 100mm, 16 kg/m3", [0.43, 0.86, 1.00, 1.00, 1.00, 1.00], "mineral wool and foams"),
    Material("Glass wool 25mm, 24 kg/m3", [0.11, 0.32, 0.56, 0.77, 0.89, 0.91], "mineral wool and foams"),
    Material("Glass wool 50mm, 24 kg/m3", [0.27, 0.54, 0.94, 1.00, 0.96, 0.96], "mineral wool and foams"),
    Material("Glass wool 75mm, 24 kg/m3", [0.28, 0.79, 1.00, 1.00, 1.00, 1.00], "mineral wool and foams"),
    Material("Glass wool 100mm, 24 kg/m3", [0.46, 1.00, 1.00, 1.00, 1.00, 1.00], "mineral wool and foams"),
    Material("Glass wool 50mm, 33 kg/m3", [0.20, 0.55, 1.00, 1.00, 1.00, 1.00], "mineral wool and foams"),
    Material("Glass wool 75mm, 33 kg/m3", [0.37, 0.85, 1.00, 1.00, 1.00, 1.00], "mineral wool and foams"),
    Material("Glass wool 100mm, 33 kg/m3", [0.53, 0.92, 1.00, 1.00, 1.00, 1.00], "mineral wool and foams"),
    Material("Glass wool 50mm, 48 kg/m3", [0.30, 0.80, 1.00, 1.00, 1.00, 1.00], "mineral wool and foams"),
    Material("Glass wool 75mm, 48 kg/m3", [0.43, 0.97, 1.00, 1.00, 1.00, 1.00], "mineral wool and foams"),
    Material("Glass wool 100mm, 48 kg/m3", [0.65, 1.00, 1.00, 1.00, 1.00, 1.00], "mineral wool and foams"),
    Material("Rock wool 50mm, 33 kg/m3 direct to masonry", [0.15, 0.60, 0.90, 0.90, 0.90, 0.85], "mineral wool and foams"),
    Material("Rock wool 100mm, 33 kg/m3 direct to masonry", [0.35, 0.95, 0.98, 0.92, 0.90, 0.85], "mineral wool and foams"),
    Material("Rock wool 50mm, 60 kg/m3 direct to masonry", [0.11, 0.60, 0.96, 0.94, 0.92, 0.82], "mineral wool and foams"),
    Material("Rock wool 75mm, 60 kg/m3 direct to masonry", [0.34, 0.95, 0.98, 0.82, 0.87, 0.86], "mineral wool and foams"),
    Material("Rock wool 30mm, 100 kg/m3 direct to masonry", [0.10, 0.40, 0.80, 0.90, 0.90, 0.90], "mineral wool and foams"),
    Material("Rock wool 30mm, 200 kg/m3 over 300mm air gap", [0.40, 0.75, 0.90, 0.80, 0.90, 0.85], "mineral wool and foams"),
    Material("Glass wool or mineral wool on solid backing 25mm", [0.20, 0.00, 0.70, 0.00, 0.90, 0.80], "mineral wool and foams"),
    Material("Glass wool or mineral wool on solid backing 50mm", [0.30, 0.00, 0.30, 0.00, 0.95, 0.90], "mineral wool and foams"),
    Material("Glass wool or mineral wool over air space on solid backing 25mm", [0.40, 0.00, 0.80, 0.00, 0.90, 0.80], "mineral wool and foams"),
    Material("Fibreglass super fine mat 50mm", [0.15, 0.40, 0.75, 0.85, 0.80, 0.85], "mineral wool and foams"),
    Material("Fibreglass scrim-covered sewn sheet 40mm", [0.40, 0.80, 0.95, 0.95, 0.80, 0.85], "mineral wool and foams"),
    Material("Fibreglass bitumen bonded mat 25mm", [0.10, 0.35, 0.50, 0.55, 0.70, 0.70], "mineral wool and foams"),
    Material("Fibreglass bitumen bonded mat 50mm", [0.30, 0.55, 0.80, 0.85, 0.75, 0.80], "mineral wool and foams"),
    Material("Fibreglass resin-bonded mat 25mm", [0.10, 0.35, 0.55, 0.65, 0.75, 0.80], "mineral wool and foams"),
    Material("Fibreglass resin-bonded mat 50mm", [0.20, 0.50, 0.70, 0.80, 0.75, 0.80], "mineral wool and foams"),
    Material("Fibreglass resin-bonded board 25mm", [0.10, 0.25, 0.55, 0.70, 0.80, 0.85], "mineral wool and foams"),
    Material("Flexible polyurethane foam 50mm", [0.25, 0.50, 0.85, 0.95, 0.90, 0.90], "mineral wool and foams"),
    Material("Rigid polyurethane foam 50mm", [0.20, 0.40, 0.65, 0.55, 0.70, 0.70], "mineral wool and foams"),
    Material("12mm expanded polystyrene on 45mm battens", [0.05, 0.15, 0.40, 0.35, 0.20, 0.20], "mineral wool and foams"),
    Material("25mm expanded polystyrene on 50mm battens", [0.10, 0.25, 0.55, 0.20, 0.10, 0.15], "mineral wool and foams"),

]

wall_treatments_and_constructions = [
    Material("Cork tiles 25mm on solid backing", [0.05, 0.10, 0.20, 0.55, 0.60, 0.55], "wall treatments and constructions"),
    Material("Cork board, 25mm on solid backing", [0.03, 0.05, 0.17, 0.52, 0.50, 0.52], "wall treatments and constructions"),
    Material("Cork board, 25mm, 2.9kg/m2, on battens", [0.15, 0.40, 0.65, 0.35, 0.35, 0.30], "wall treatments and constructions"),
    Material("Glass blocks or glazed tiles as wall finish", [0.01, 0.00, 0.01, 0.00, 0.01, 0.01], "wall treatments and constructions"),
    Material("Muslin covered cotton felt 25mm", [0.15, 0.45, 0.70, 0.85, 0.95, 0.85], "wall treatments and constructions"),
    Material("Pin up boarding- medium hardboard on solid backing", [0.05, 0.00, 0.10, 0.00, 0.10, 0.10], "wall treatments and constructions"),
    Material("Pin up boarding- softboard on solid backing", [0.05, 0.00, 0.10, 0.00, 0.10, 0.10], "wall treatments and constructions"),
    Material("Fibreboard on solid backing 12mm", [0.05, 0.10, 0.15, 0.25, 0.30, 0.30], "wall treatments and constructions"),
    Material("25mm thick hair felt, covered by scrim cloth on solid backing", [0.10, 0.00, 0.70, 0.00, 0.80, 0.80], "wall treatments and constructions"),
    Material("Fibreboard on solid backing", [0.05, 0.00, 0.15, 0.00, 0.30, 0.30], "wall treatments and constructions"),
    Material("Fibreboard on solid backing - painted", [0.05, 0.00, 0.10, 0.00, 0.15, 0.15], "wall treatments and constructions"),
    Material("Fibreboard over airspace on solid wall", [0.30, 0.00, 0.30, 0.00, 0.30, 0.30], "wall treatments and constructions"),
    Material("Fibreboard over airspace on solid wall - painted", [0.30, 0.00, 0.15, 0.00, 0.10, 0.10], "wall treatments and constructions"),
    Material("Plaster on lath, deep air space", [0.20, 0.15, 0.10, 0.05, 0.05, 0.05], "wall treatments and constructions"),
    Material("Plaster decorative panels, walls", [0.20, 0.15, 0.10, 0.08, 0.04, 0.02], "wall treatments and constructions"),
    Material("Acoustic plaster to solid backing 25mm", [0.03, 0.15, 0.50, 0.80, 0.85, 0.80], "wall treatments and constructions"),
    Material("9mm acoustic plaster to solid backing", [0.02, 0.08, 0.30, 0.60, 0.80, 0.90], "wall treatments and constructions"),
    Material("9mm acoustic plaster on plasterboard, 75mm airspace", [0.30, 0.30, 0.60, 0.80, 0.75, 0.75], "wall treatments and constructions"),
    Material("12.5mm acoustic plaster on plaster backing over 75mm air space", [0.35, 0.35, 0.40, 0.55, 0.70, 0.70], "wall treatments and constructions"),
    Material("Woodwool slabs, unplastered on solid backing 25mm", [0.10, 0.00, 0.40, 0.00, 0.60, 0.60], "wall treatments and constructions"),
    Material("Woodwool slabs, unplastered on solid backing 50mm", [0.10, 0.20, 0.45, 0.80, 0.60, 0.75], "wall treatments and constructions"),
    Material("Woodwool slabs, unplastered on solid backing 75mm", [0.20, 0.00, 0.80, 0.00, 0.80, 0.80], "wall treatments and constructions"),
    Material("Woodwool slabs, unplastered over 20mm airspace on solid backing 25mm", [0.15, 0.00, 0.60, 0.00, 0.60, 0.70], "wall treatments and constructions"),
    Material("Plasterboard backed with 25mm thick bitumen-bonded fibreglass on 50mm battens", [0.30, 0.20, 0.15, 0.05, 0.05, 0.05], "wall treatments and constructions"),
    Material("Curtains hung in folds against soild wall", [0.05, 0.15, 0.35, 0.40, 0.50, 0.50], "wall treatments and constructions"),
    Material("Cotton Curtains (0.5kg/m2),draped to 75% area approx. 130mm from wall", [0.30, 0.45, 0.65, 0.56, 0.59, 0.71], "wall treatments and constructions"),
    Material("Lightweight curtains (0.2 kg/m2) hung 90mm from wall", [0.05, 0.06, 0.39, 0.63, 0.70, 0.73], "wall treatments and constructions"),
    Material("Curtains of close-woven glass mat hung 50mm from wall", [0.03, 0.03, 0.15, 0.40, 0.50, 0.50], "wall treatments and constructions"),
    Material("Curtains, medium velour, 50% gather, over soild backing", [0.05, 0.25, 0.40, 0.50, 0.60, 0.50], "wall treatments and constructions"),
    Material("Curtains (medium fabrics) hung straight and close to wall", [0.05, 0.00, 0.25, 0.00, 0.30, 0.40], "wall treatments and constructions"),
    Material("Curtains in folds against wall", [0.05, 0.15, 0.35, 0.40, 0.50, 0.50], "wall treatments and constructions"),
    Material("Curtains ( medium fabrics ) double widths in folds spaced away from wall", [0.10, 0.00, 0.40, 0.00, 0.50, 0.60], "wall treatments and constructions"),
    Material("Acoustic banner, 0.5 kg/m2 wool serge, 100mm from wall", [0.11, 0.40, 0.70, 0.74, 0.88, 0.89], "wall treatments and constructions"),
]

floors = [
    Material("Smooth marble or terrazzo slabs", [0.01, 0.01, 0.01, 0.01, 0.02, 0.02], "floors"),
    Material("Raised computer floor, steel-faced 45mm chipboard 800mm above concrete floor, no carpet", [0.08, 0.07, 0.06, 0.07, 0.08, 0.08], "floors"),
    Material("Raised computer floor, steel-faced 45mm chipboard 800mm above concrete floor, office-grade carpet tiles", [0.27, 0.26, 0.52, 0.43, 0.51, 0.58], "floors"),
    Material("Wooden floor on joists", [0.15, 0.11, 0.10, 0.07, 0.06, 0.07], "floors"),
    Material("Parquet fixed in asphalt, on concrete", [0.04, 0.04, 0.07, 0.06, 0.06, 0.07], "floors"),
    Material("Parquet on counterfloor", [0.20, 0.15, 0.10, 0.10, 0.05, 0.10], "floors"),
    Material("Linoleum or vinyl stuck to concrete", [0.02, 0.02, 0.03, 0.04, 0.04, 0.05], "floors"),
    Material("Layer of rubber, cork, linoleum + underlay, or vinyl+underlay stuck to concrete", [0.02, 0.02, 0.04, 0.05, 0.05, 0.10], "floors"),
    Material("5mm needle-felt stuck to concrete", [0.01, 0.02, 0.05, 0.15, 0.30, 0.40], "floors"),
    Material("6mm pile carpet bonded to closed-cell foam underlay", [0.03, 0.09, 0.25, 0.31, 0.33, 0.44], "floors"),
    Material("6mm pile carpet bonded to open-cell foam underlay", [0.03, 0.09, 0.25, 0.31, 0.33, 0.44], "floors"),
    Material("9mm pile carpet, tufted on felt underlay", [0.08, 0.08, 0.30, 0.60, 0.75, 0.80], "floors"),
    Material("Composition flooring", [0.05, 0.05, 0.05, 0.05, 0.05, 0.05], "floors"),
    Material("Haircord carpet on felt underlay 6mm", [0.05, 0.05, 0.10, 0.20, 0.45, 0.65], "floors"),
    Material("Medium pile carpet on sponge rubber underlay 10mm", [0.50, 0.10, 0.30, 0.50, 0.65, 0.70], "floors"),
    Material("Thick pile carpet on sponge rubber underlay 15mm", [0.15, 0.25, 0.50, 0.60, 0.70, 0.70], "floors"),
    Material("Rubber floor tiles 6mm", [0.05, 0.05, 0.10, 0.10, 0.05, 0.05], "floors"),
    Material("Carpet, thin, over thin felt on concrete", [0.10, 0.15, 0.25, 0.30, 0.30, 0.30], "floors"),
    Material("Carpet, thin, over thin felt on wood floor", [0.20, 0.25, 0.30, 0.30, 0.30, 0.30], "floors"),
    Material("Carpet, needlepunch 5mm", [0.03, 0.05, 0.05, 0.25, 0.35, 0.50], "floors"),
    Material("Stone floor, plain or tooled or granolithic finish", [0.02, 0.00, 0.02, 0.00, 0.05, 0.05], "floors"),
    Material("Corkfloor tiles 14mm", [0.00, 0.05, 0.15, 0.25, 0.25, 0.00], "floors"),
    Material("Sheet rubber (hard) 6mm", [0.00, 0.05, 0.05, 0.10, 0.05, 0.00], "floors"),
    Material("Woodblock/linoleum/rubber/cork tiles (thin) on solid floor (or wall)", [0.02, 0.04, 0.05, 0.05, 0.10, 0.05], "floors"),
    Material("Floor tiles, plastic or linoleum", [0.03, 0.00, 0.03, 0.00, 0.05, 0.05], "floors"),
    Material("Steel decking", [0.13, 0.09, 0.08, 0.09, 0.11, 0.11], "floors"),

]
