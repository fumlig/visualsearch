import numpy as np
from gym_search.utils import lerp

def pick_color(c, palette):
    c = np.array(c)
    c /= c.max()
    i = lerp(0, len(palette)-1, c).astype(int)
    p = np.array(palette, dtype=np.uint8)
    return p[i]


def add_with_alpha(bg, fg, alpha):
    bg = np.array(bg, dtype=np.uint8)
    fg = np.array(fg, dtype=np.uint8)

    return (1-alpha)*bg + alpha*fg


BLUE_MARBLE = [
    (0, 0, 0),
    (0, 0, 46),
    (0, 0, 58),
    (0, 0, 69),
    (0, 0, 81),
    (0, 0, 92),
    (0, 0, 104),
    (0, 0, 116),
    (0, 3, 116),
    (1, 6, 116),
    (2, 8, 116),
    (2, 11, 116),
    (3, 13, 117),
    (4, 16, 117),
    (5, 18, 117),
    (5, 21, 117),
    (6, 23, 117),
    (7, 26, 118),
    (8, 28, 118),
    (8, 31, 118),
    (9, 33, 118),
    (10, 36, 118),
    (11, 38, 119),
    (11, 41, 119),
    (12, 43, 119),
    (13, 45, 119),
    (14, 48, 119),
    (15, 50, 120),
    (15, 52, 120),
    (16, 55, 120),
    (17, 57, 120),
    (18, 59, 120),
    (18, 61, 121),
    (19, 64, 121),
    (20, 66, 121),
    (21, 68, 121),
    (22, 70, 121),
    (22, 72, 122),
    (23, 74, 122),
    (24, 77, 122),
    (25, 79, 122),
    (26, 81, 122),
    (26, 83, 123),
    (27, 85, 123),
    (28, 87, 123),
    (29, 89, 123),
    (30, 91, 123),
    (31, 93, 124),
    (31, 95, 124),
    (32, 97, 124),
    (33, 99, 124),
    (34, 100, 124),
    (35, 102, 125),
    (36, 104, 125),
    (36, 106, 125),
    (37, 108, 125),
    (38, 109, 125),
    (39, 111, 126),
    (40, 113, 126),
    (41, 115, 126),
    (41, 116, 126),
    (42, 118, 126),
    (43, 120, 127),
    (44, 121, 127),
    (45, 123, 127),
    (46, 125, 127),
    (47, 126, 127),
    (48, 128, 128),
    (48, 128, 126),
    (48, 129, 125),
    (49, 129, 124),
    (49, 130, 123),
    (50, 131, 122),
    (50, 131, 120),
    (51, 132, 119),
    (51, 133, 118),
    (52, 133, 117),
    (52, 134, 115),
    (53, 134, 114),
    (53, 135, 113),
    (54, 136, 111),
    (54, 136, 110),
    (55, 137, 109),
    (55, 138, 108),
    (56, 138, 106),
    (56, 139, 105),
    (57, 140, 104),
    (57, 140, 102),
    (58, 141, 101),
    (58, 141, 100),
    (59, 142,  98),
    (59, 143,  97),
    (60, 143,  96),
    (61, 144,  94),
    (61, 145,  93),
    (62, 145,  92),
    (62, 146,  90),
    (63, 146,  89),
    (63, 147,  88),
    (64, 148,  86),
    (64, 148,  85),
    (65, 149,  84),
    (65, 150,  82),
    (66, 150,  81),
    (67, 151,  80),
    (67, 151,  78),
    (68, 152,  77),
    (68, 153,  76),
    (69, 153,  74),
    (69, 154,  73),
    (70, 155,  71),
    (71, 155,  70),
    (73, 156,  71),
    (76, 156,  72),
    (78, 157,  72),
    (81, 158,  73),
    (83, 158,  73),
    (86, 159,  74),
    (88, 160,  75),
    (91, 160,  75),
    (94, 161,  76),
    (96, 161,  76),
    (99, 162,  77),
    (101, 163, 77),
    (104, 163, 78),
    (106, 164, 79),
    (109, 165, 79),
    (111, 165, 80),
    (114, 166, 80),
    (117, 166, 81),
    (119, 167, 82),
    (121, 168, 82),
    (122, 168, 82),
    (124, 168, 83),
    (126, 169, 83),
    (128, 169, 83),
    (129, 170, 84),
    (131, 170, 84),
    (133, 171, 84),
    (135, 171, 85),
    (136, 172, 85),
    (138, 172, 85),
    (140, 172, 86),
    (141, 173, 86),
    (143, 173, 86),
    (145, 174, 87),
    (147, 174, 87),
    (149, 175, 87),
    (150, 175, 88),
    (152, 175, 88),
    (154, 176, 88),
    (156, 176, 89),
    (157, 177, 89),
    (159, 177, 89),
    (161, 178, 90),
    (163, 178, 90),
    (165, 179, 90),
    (166, 179, 91),
    (168, 179, 91),
    (170, 180, 91),
    (172, 180, 92),
    (174, 181, 92),
    (175, 181, 92),
    (177, 182, 93),
    (179, 182, 93),
    (181, 183, 93),
    (183, 183, 94),
    (183, 182, 94),
    (184, 181, 94),
    (184, 181, 95),
    (185, 180, 95),
    (185, 179, 95),
    (186, 178, 96),
    (186, 177, 96),
    (187, 176, 97),
    (187, 175, 97),
    (187, 174, 97),
    (188, 173, 98),
    (188, 172,98),
    (189, 171, 98),
    (189, 170, 99),
    (190, 169, 99),
    (190, 168, 99),
    (190, 167, 100),
    (191, 166, 100),
    (191, 165, 100),
    (192, 164, 101),
    (192, 163, 101),
    (193, 163, 104),
    (195, 164, 106),
    (196, 164, 108),
    (197, 165, 111),
    (198, 165, 113),
    (199, 166, 116),
    (201, 167, 118),
    (202, 167, 121),
    (203, 168, 123),
    (204, 169, 126),
    (205, 170, 129),
    (207, 171, 131),
    (208, 172, 134),
    (209, 173, 137),
    (210, 174, 139),
    (211, 175, 142),
    (213, 176, 145),
    (214, 177, 148),
    (215, 178, 150),
    (216, 179, 153),
    (217, 181, 156),
    (219, 182, 159),
    (220, 184, 162),
    (221, 185, 165),
    (222, 187, 168),
    (223, 188, 170),
    (225, 190, 173),
    (226, 192, 176),
    (227, 194, 179),
    (228, 196, 182),
    (229, 198, 185),
    (231, 200, 189),
    (232, 202, 192),
    (233, 204, 195),
    (234, 206, 198),
    (235, 208, 201),
    (237, 211, 204),
    (238, 213, 207),
    (239, 215, 211),
    (240, 218, 214),
    (241, 221, 217),
    (243, 223, 220),
    (244, 226, 224),
    (245, 229, 227),
    (246, 232, 230),
    (247, 235, 234),
    (249, 238, 237),
    (250, 241, 241),
    (251, 244, 244),
    (252, 248, 248),
    (253, 251, 251),
    (255, 255, 255),
]

EARTH_TOON = [BLUE_MARBLE[i] for i in range(25, len(BLUE_MARBLE), 5)]
