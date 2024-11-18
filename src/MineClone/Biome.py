from dataclasses import dataclass
from scipy.interpolate import UnivariateSpline

import nbtlib

from POGLE.Core.Core import Renum
from Generation import BiomeParams

_last_biome_id: int = -1


def _bid():
    global _last_biome_id
    _last_biome_id += 1
    return _last_biome_id


class BiomeID(Renum):
    Null = -1
    # Cave Biomes
    #############
    # Caves
    CavesDripstone = _bid()
    CavesLush = _bid()

    # Deep Dark
    DeepDark = _bid()

    # Non-Inland
    ############
    # Mushroom
    MushroomFields = _bid()

    # Deep Ocean
    OceanDeepFrozen = _bid()
    OceanDeepCold = _bid()
    OceanDeepLukewarm = _bid()
    OceanDeep = _bid()

    # Non-Deep Ocean
    OceanFrozen = _bid()
    OceanCold = _bid()
    Ocean = _bid()
    OceanLukewarm = _bid()

    # Warm Ocean
    OceanWarm = _bid()

    # Inland Surface
    ################
    # River
    River = _bid()
    RiverFrozen = _bid()

    # Beach
    StonyShore = _bid()
    Beach = _bid()
    BeachSnowy = _bid()

    # Desert
    Desert = _bid()

    # Savanna
    Savanna = _bid()
    SavannaWindsept = _bid()
    SavannaPlateau = _bid()

    # Swamp
    Swamp = _bid()
    SwampMangrove = _bid()

    # Snowy
    SnowySlope = _bid()
    Grove = _bid()
    IceSpikes = _bid()

    # Badlands
    Badlands = _bid()
    BadlandsEroded = _bid()
    BadlandsWooded = _bid()

    # Plains
    Plains = _bid()
    PlainsSnowy = _bid()
    PlainsSunflower = _bid()

    # Forest
    Forest = _bid()
    ForestFlower = _bid()
    ForestBirch = _bid()
    ForestOldGrowthBirch = _bid()
    ForestDark = _bid()
    ForestWindswept = _bid()

    # Taiga
    Taiga = _bid()
    TaigaSnowy = _bid()
    TaigaOldGrowthSpruce = _bid()
    TaigaOldGrowthPine = _bid()

    # Jungle
    Jungle = _bid()
    JungleSparse = _bid()
    JungleBamboo = _bid()

    # Meadow
    Meadow = _bid()

    # Cherry Grove
    CherryGrove = _bid()

    # Pale Garden
    PaleGarden = _bid()

    # Peaks
    PeaksJagged = _bid()
    PeaksFrozen = _bid()
    PeaksStony = _bid()

    # Hills
    HillsWindswept = _bid()
    HillsWindsweptGravelly = _bid()


def get_shattered_biome_id(biome_params: BiomeParams) -> BiomeID:
    match biome_params.level_humidty:
        case 0 | 1:
            match biome_params.level_temperature:
                case 0 | 1:
                    return BiomeID.HillsWindsweptGravelly
                case 2:
                    return BiomeID.HillsWindswept
                case 3:
                    return BiomeID.Savanna
                case 4:
                    return BiomeID.Desert
        case 2:
            match biome_params.level_temperature:
                case 0 | 1 | 2:
                    return BiomeID.HillsWindswept
                case 3:
                    if biome_params.weirdness < 0:
                        return BiomeID.Savanna
                    elif biome_params.weirdness > 0:
                        return BiomeID.Plains
                case 4:
                    return BiomeID.Desert
        case 3:
            match biome_params.level_temperature:
                case 0 | 1 | 2:
                    return BiomeID.ForestWindswept
                case 3:
                    if biome_params.weirdness < 0:
                        return BiomeID.Jungle
                    elif biome_params.weirdness > 0:
                        return BiomeID.JungleSparse
                case 4:
                    return BiomeID.Desert
        case 4:
            match biome_params.level_temperature:
                case 0 | 1 | 2:
                    return BiomeID.ForestWindswept
                case 3:
                    if biome_params.weirdness < 0:
                        return BiomeID.Jungle
                    elif biome_params.weirdness > 0:
                        return BiomeID.JungleBamboo
                case 4:
                    return BiomeID.Desert
    return BiomeID.Null


def get_plateau_biome_id(biome_params: BiomeParams) -> BiomeID:
    match biome_params.level_humidty:
        case 0:
            match biome_params.level_temperature:
                case 0:
                    if biome_params.weirdness < 0:
                        return BiomeID.PlainsSnowy
                    elif biome_params.weirdness > 0:
                        return BiomeID.IceSpikes
                case 1 | 2:
                    return BiomeID.PlainsSnowy
                case 3 | 4:
                    return BiomeID.TaigaSnowy
        case 1:
            match biome_params.level_temperature:
                case 0:
                    if biome_params.weirdness < 0:
                        return BiomeID.Meadow
                    elif biome_params.weirdness > 0:
                        return BiomeID.CherryGrove
                case 1:
                    return BiomeID.Meadow
                case 2:
                    if biome_params.weirdness < 0:
                        return BiomeID.Forest
                    elif biome_params.weirdness > 0:
                        return BiomeID.Meadow
                case 3:
                    if biome_params.weirdness < 0:
                        return BiomeID.Taiga
                    elif biome_params.weirdness > 0:
                        return BiomeID.Meadow
                case 4:
                    if biome_params.weirdness < 0:
                        return BiomeID.TaigaOldGrowthSpruce
                    elif biome_params.weirdness > 0:
                        return BiomeID.TaigaOldGrowthPine
        case 2:
            match biome_params.level_temperature:
                case 0 | 1:
                    if biome_params.weirdness < 0:
                        return BiomeID.Meadow
                    elif biome_params.weirdness > 0:
                        return BiomeID.CherryGrove
                case 2:
                    if biome_params.weirdness < 0:
                        return BiomeID.Meadow
                    elif biome_params.weirdness > 0:
                        return BiomeID.Forest
                case 3:
                    if biome_params.weirdness < 0:
                        return BiomeID.Meadow
                    elif biome_params.weirdness > 0:
                        return BiomeID.ForestBirch
                case 4:
                    if biome_params.weirdness < 0:
                        return BiomeID.ForestDark
                    elif biome_params.weirdness > 0:
                        return BiomeID.PaleGarden
        case 3:
            match biome_params.level_temperature:
                case 0 | 1:
                    return BiomeID.SavannaPlateau
                case 2 | 3:
                    return BiomeID.Forest
                case 4:
                    return BiomeID.Jungle
        case 4:
            match biome_params.level_temperature:
                case 0 | 1:
                    if biome_params.weirdness < 0:
                        return BiomeID.Badlands
                    elif biome_params.weirdness > 0:
                        return BiomeID.BadlandsEroded
                case 2:
                    return BiomeID.Badlands
                case 3 | 4:
                    return BiomeID.BadlandsWooded
    return BiomeID.Null


def get_middle_biome_id(biome_params: BiomeParams) -> BiomeID:
    match biome_params.level_humidty:
        case 0:
            match biome_params.level_temperature:
                case 0:
                    if biome_params.weirdness < 0:
                        return BiomeID.PlainsSnowy
                    elif biome_params.weirdness > 0:
                        return BiomeID.IceSpikes
                case 1:
                    return BiomeID.Plains
                case 2:
                    if biome_params.weirdness < 0:
                        return BiomeID.ForestFlower
                    elif biome_params.weirdness > 0:
                        return BiomeID.PlainsSunflower
                case 3:
                    return BiomeID.Savanna
                case 4:
                    return BiomeID.Desert
        case 1:
            match biome_params.level_temperature:
                case 0:
                    return BiomeID.PlainsSnowy
                case 1 | 2:
                    return BiomeID.Plains
                case 3:
                    return BiomeID.Savanna
                case 4:
                    return BiomeID.Desert
        case 2:
            match biome_params.level_temperature:
                case 0:
                    if biome_params.weirdness < 0:
                        return BiomeID.PlainsSnowy
                    elif biome_params.weirdness > 0:
                        return BiomeID.TaigaSnowy
                case 1 | 2:
                    return BiomeID.Forest
                case 3:
                    if biome_params.weirdness < 0:
                        return BiomeID.Forest
                    elif biome_params.weirdness > 0:
                        return BiomeID.Plains
                case 4:
                    return BiomeID.Desert
        case 3:
            match biome_params.level_temperature:
                case 0:
                    return BiomeID.TaigaSnowy
                case 1:
                    return BiomeID.Taiga
                case 2:
                    if biome_params.weirdness < 0:
                        return BiomeID.ForestBirch
                    elif biome_params.weirdness > 0:
                        return BiomeID.ForestOldGrowthBirch
                case 3:
                    if biome_params.weirdness < 0:
                        return BiomeID.Jungle
                    elif biome_params.weirdness > 0:
                        return BiomeID.JungleSparse
                case 4:
                    return BiomeID.Desert
        case 4:
            match biome_params.level_temperature:
                case 0:
                    return BiomeID.Taiga
                case 1:
                    if biome_params.weirdness < 0:
                        return BiomeID.TaigaOldGrowthSpruce
                    elif biome_params.weirdness > 0:
                        return BiomeID.TaigaOldGrowthPine
                case 2:
                    return BiomeID.ForestDark
                case 3:
                    if biome_params.weirdness < 0:
                        return BiomeID.Jungle
                    elif biome_params.weirdness > 0:
                        return BiomeID.JungleSparse
                case 4:
                    return BiomeID.Desert
    return BiomeID.Null


def get_badland_biome_id(biome_params: BiomeParams) -> BiomeID:
    match biome_params.level_temperature:
        case 0 | 1:
            if biome_params.weirdness < 0:
                return BiomeID.Badlands
            elif biome_params.weirdness > 0:
                return BiomeID.BadlandsEroded
        case 2:
            return BiomeID.Badlands
        case 3 | 4:
            return BiomeID.BadlandsWooded
    return BiomeID.Null


def get_beach_biome_id(biome_params: BiomeParams) -> BiomeID:
    match biome_params.level_temperature:
        case 0:
            return BiomeID.BeachSnowy
        case 1 | 2 | 3:
            return BiomeID.Beach
        case 4:
            return BiomeID.Desert
    return BiomeID.Null


def get_surface_biome_id(biome_params: BiomeParams) -> BiomeID:
    match biome_params.peak_valley:
        case PV.Valleys:
            match biome_params.continent:
                case Continent.Coast:
                    if biome_params.level_temperature == 0:
                        return BiomeID.RiverFrozen
                    else:
                        return BiomeID.River
                case Continent.NearInland:
                    match biome_params.level_erosion:
                        case 0 | 1 | 2 | 3 | 4 | 5:
                            if biome_params.level_temperature == 0:
                                return BiomeID.RiverFrozen
                            else:
                                return BiomeID.River
                        case 6:
                            match biome_params.level_temperature:
                                case 0:
                                    return BiomeID.RiverFrozen
                                case 1 | 2:
                                    return BiomeID.Swamp
                                case 3 | 4:
                                    return BiomeID.SwampMangrove
                case Continent.MidInland:
                    match biome_params.level_erosion:
                        case 0 | 1:
                            if biome_params.level_temperature == 4:
                                return get_badland_biome_id(biome_params)
                            else:
                                return get_middle_biome_id(biome_params)
                        case 2 | 3 | 4 | 5:
                            if biome_params.level_temperature == 0:
                                return BiomeID.RiverFrozen
                            else:
                                return BiomeID.River
                        case 6:
                            match biome_params.level_temperature:
                                case 0:
                                    return BiomeID.RiverFrozen
                                case 1 | 2:
                                    return BiomeID.Swamp
                                case 3 | 4:
                                    return BiomeID.SwampMangrove

                case Continent.FarInland:
                    match biome_params.level_erosion:
                        case 0 | 1:
                            if biome_params.level_temperature == 4:
                                return get_badland_biome_id(biome_params)
                            else:
                                return get_middle_biome_id(biome_params)
                        case 2 | 3 | 4 | 5:
                            if biome_params.level_temperature == 0:
                                return BiomeID.RiverFrozen
                            else:
                                return BiomeID.River
                        case 6:
                            match biome_params.level_temperature:
                                case 0:
                                    return BiomeID.RiverFrozen
                                case 1 | 2:
                                    return BiomeID.Swamp
                                case 3 | 4:
                                    return BiomeID.SwampMangrove
        case PV.Low:
            match biome_params.continent:
                case Continent.Coast:
                    match biome_params.level_erosion:
                        case 0 | 1 | 2:
                            return BiomeID.StonyShore
                        case 3 | 4 | 6:
                            return get_beach_biome_id(biome_params)
                        case 5:
                            if biome_params.weirdness < 0:
                                return get_beach_biome_id(biome_params)
                            elif biome_params.weirdness > 0:
                                if biome_params.level_temperature < 2 or biome_params.level_humidty == 4:
                                    return BiomeID.SavannaWindsept
                                elif biome_params.level_temperature > 1 and biome_params.level_humidty < 4:
                                    return get_badland_biome_id(biome_params)
                case Continent.NearInland:
                    match biome_params.level_erosion:
                        case 0 | 1:
                            if biome_params.level_temperature == 4:
                                return get_badland_biome_id(biome_params)
                            else:
                                return get_middle_biome_id(biome_params)
                        case 2 | 3 | 4:
                            return get_middle_biome_id(biome_params)
                        case 5:
                            if biome_params.weirdness < 0 or biome_params.level_temperature < 2 or biome_params.level_humidty == 4:
                                return get_middle_biome_id(biome_params)
                            elif biome_params.weirdness > 0 and biome_params.temperature > 1 and biome_params.humidity < 4:
                                return BiomeID.SavannaWindsept
                        case 6:
                            match biome_params.level_temperature:
                                case 0:
                                    return get_middle_biome_id(biome_params)
                                case 1 | 2:
                                    return BiomeID.Swamp
                                case 3 | 4:
                                    return BiomeID.SwampMangrove
                case Continent.MidInland:
                    match biome_params.level_erosion:
                        case 0 | 1:
                            match biome_params.level_temperature:
                                case 0:
                                    if biome_params.level_humidty < 2:
                                        return BiomeID.SnowySlope
                                    else:
                                        return BiomeID.Grove
                                case 1 | 2 | 3:
                                    return get_middle_biome_id(biome_params)
                                case 4:
                                    return get_badland_biome_id(biome_params)
                        case 2 | 3:
                            if biome_params.level_temperature < 4:
                                return get_middle_biome_id(biome_params)
                            elif biome_params.level_temperature == 4:
                                return get_badland_biome_id(biome_params)
                        case 4 | 5:
                            return get_middle_biome_id(biome_params)
                        case 6:
                            match biome_params.level_temperature:
                                case 0:
                                    return get_middle_biome_id(biome_params)
                                case 1 | 2:
                                    return BiomeID.Swamp
                                case 3 | 4:
                                    return BiomeID.SwampMangrove
                case Continent.FarInland:
                    match biome_params.level_erosion:
                        case 0 | 1:
                            match biome_params.level_temperature:
                                case 0:
                                    if biome_params.level_humidty < 2:
                                        return BiomeID.SnowySlope
                                    else:
                                        return BiomeID.Grove
                                case 1 | 2 | 3:
                                    return get_middle_biome_id(biome_params)
                                case 4:
                                    return get_badland_biome_id(biome_params)
                        case 2 | 3:
                            if biome_params.level_temperature < 4:
                                return get_middle_biome_id(biome_params)
                            elif biome_params.level_temperature == 4:
                                return get_badland_biome_id(biome_params)
                        case 4 | 5:
                            return get_middle_biome_id(biome_params)
                        case 6:
                            match biome_params.level_temperature:
                                case 0:
                                    return get_middle_biome_id(biome_params)
                                case 1 | 2:
                                    return BiomeID.Swamp
                                case 3 | 4:
                                    return BiomeID.SwampMangrove
        case PV.Mid:
            match biome_params.continent:
                case Continent.Coast:
                    match biome_params.level_erosion:
                        case 0 | 1 | 2:
                            return BiomeID.StonyShore
                        case 3:
                            return get_middle_biome_id(biome_params)
                        case 4 | 6:
                            if biome_params.weirdness < 0:
                                return get_beach_biome_id(biome_params)
                            elif biome_params.weirdness > 0:
                                return get_middle_biome_id(biome_params)
                        case 5:
                            if biome_params.weirdness < 0:
                                return get_beach_biome_id(biome_params)
                            elif biome_params.weirdness > 0:
                                if biome_params.level_temperature < 2 or biome_params.level_humidty == 4:
                                    return get_middle_biome_id(biome_params)
                                elif biome_params.level_temperature > 1 and biome_params.level_humidty < 4:
                                    return BiomeID.SavannaWindsept

                case Continent.NearInland:
                    match biome_params.level_erosion:
                        case 0:
                            if biome_params.level_temperature < 3:
                                if biome_params.level_humidty < 2:
                                    return BiomeID.SnowySlope
                                else:
                                    return BiomeID.Grove
                            else:
                                return get_plateau_biome_id(biome_params)
                        case 1:
                            if biome_params.level_temperature == 0:
                                if biome_params.level_humidty < 2:
                                    return BiomeID.SnowySlope
                                else:
                                    return BiomeID.Grove
                            elif 0 < biome_params.level_temperature < 4:
                                return get_middle_biome_id(biome_params)
                            else:
                                return get_badland_biome_id(biome_params)
                        case 2 | 3 | 4:
                            return get_middle_biome_id(biome_params)
                        case 5:
                            if biome_params.weirdness < 0 or biome_params.level_temperature < 2 or biome_params.level_humidty == 4:
                                return get_middle_biome_id(biome_params)
                            if biome_params.weirdness > 0 and biome_params.level_temperature > 1 and biome_params.level_humidty < 4:
                                return BiomeID.SavannaWindsept
                        case 6:
                            match biome_params.level_temperature:
                                case 0:
                                    return get_middle_biome_id(biome_params)
                                case 1 | 2:
                                    return BiomeID.Swamp
                                case 3 | 4:
                                    return BiomeID.SwampMangrove
                case Continent.MidInland:
                    match biome_params.level_erosion:
                        case 0:
                            if biome_params.level_temperature < 3:
                                if biome_params.level_humidty < 2:
                                    return BiomeID.SnowySlope
                                else:
                                    return BiomeID.Grove
                            else:
                                return get_plateau_biome_id(biome_params)
                        case 1:
                            if biome_params.level_temperature == 0:
                                if biome_params.level_humidty < 2:
                                    return BiomeID.SnowySlope
                                else:
                                    return BiomeID.Grove
                            elif 0 < biome_params.level_temperature < 4:
                                return get_middle_biome_id(biome_params)
                            else:
                                return get_badland_biome_id(biome_params)
                        case 2 | 3:
                            if biome_params.level_temperature == 4:
                                return get_badland_biome_id(biome_params)
                            else:
                                return get_middle_biome_id(biome_params)
                        case 4:
                            return get_middle_biome_id(biome_params)
                        case 5:
                            return get_shattered_biome_id(biome_params)
                        case 6:
                            match biome_params.level_temperature:
                                case 0:
                                    return get_middle_biome_id(biome_params)
                                case 1 | 2:
                                    return BiomeID.Swamp
                                case 3 | 4:
                                    return BiomeID.SwampMangrove
                case Continent.FarInland:
                    match biome_params.level_erosion:
                        case 0:
                            if biome_params.level_temperature < 3:
                                if biome_params.level_humidty < 2:
                                    return BiomeID.SnowySlope
                                else:
                                    return BiomeID.Grove
                            else:
                                return get_plateau_biome_id(biome_params)
                        case 1:
                            if biome_params.level_temperature == 0:
                                if biome_params.level_humidty < 2:
                                    return BiomeID.SnowySlope
                                else:
                                    return BiomeID.Grove
                            else:
                                return get_badland_biome_id(biome_params)
                        case 2:
                            return get_plateau_biome_id(biome_params)
                        case 3:
                            if biome_params.level_temperature == 4:
                                return get_middle_biome_id(biome_params)
                            else:
                                return get_badland_biome_id(biome_params)
                        case 4:
                            return get_middle_biome_id(biome_params)
                        case 5:
                            return get_shattered_biome_id(biome_params)
                        case 6:
                            match biome_params.level_temperature:
                                case 0:
                                    return get_middle_biome_id(biome_params)
                                case 1 | 2:
                                    return BiomeID.Swamp
                                case 3 | 4:
                                    return BiomeID.SwampMangrove
        case PV.High:
            match biome_params.continent:
                case Continent.Coast:
                    match biome_params.level_erosion:
                        case 0 | 1 | 2 | 3 | 4:
                            return get_middle_biome_id(biome_params)
                        case 5:
                            if biome_params.weirdness < 0 or biome_params.level_temperature < 2 or biome_params.level_humidty == 4:
                                return get_middle_biome_id(biome_params)
                            elif biome_params.weirdness > 0 and biome_params.level_temperature > 1 and biome_params.level_humidty < 4:
                                return BiomeID.SavannaWindsept
                        case 6:
                            return get_middle_biome_id(biome_params)
                case Continent.NearInland:
                    match biome_params.level_erosion:
                        case 0:
                            if biome_params.level_temperature < 3 and biome_params.level_humidty < 2:
                                return BiomeID.SnowySlope
                            elif biome_params.level_temperature < 3 and biome_params.level_humidty > 1:
                                return BiomeID.Grove
                            elif biome_params.level_temperature > 2:
                                return get_plateau_biome_id(biome_params)
                        case 1:
                            if biome_params.level_temperature == 0 and biome_params.level_humidty < 2:
                                return BiomeID.SnowySlope
                            elif biome_params.level_temperature == 0 and biome_params.level_humidty > 1:
                                return BiomeID.Grove
                            elif 0 < biome_params.level_temperature < 4:
                                return get_middle_biome_id(biome_params)
                            elif biome_params.level_temperature == 4:
                                return get_badland_biome_id(biome_params)
                        case 2 | 3 | 4:
                            return get_middle_biome_id(biome_params)
                        case 5:
                            if biome_params.weirdness < 0 or biome_params.level_temperature < 2 or biome_params.level_humidty == 4:
                                return get_middle_biome_id(biome_params)
                            elif biome_params.weirdness > 0 and biome_params.level_temperature > 1 and biome_params.level_humidty < 4:
                                return BiomeID.SavannaWindsept
                        case 6:
                            return get_middle_biome_id(biome_params)
                case Continent.MidInland:
                    match biome_params.level_erosion:
                        case 0:
                            match biome_params.level_temperature:
                                case 0 | 1 | 2:
                                    if biome_params.weirdness < 0:
                                        return BiomeID.PeaksJagged
                                    elif biome_params.weirdness > 0:
                                        return BiomeID.PeaksFrozen
                                case 3:
                                    return BiomeID.PeaksStony
                                case 4:
                                    return get_badland_biome_id(biome_params)
                        case 1:
                            if biome_params.level_temperature < 3:
                                if biome_params.level_humidty < 2:
                                    return BiomeID.SnowySlope
                                else:
                                    return BiomeID.Grove
                            else:
                                return get_plateau_biome_id(biome_params)
                        case 2:
                            return get_plateau_biome_id(biome_params)
                        case 3 | 4:
                            if biome_params.level_temperature == 4:
                                return get_badland_biome_id(biome_params)
                            else:
                                return get_middle_biome_id(biome_params)
                        case 5:
                            return get_shattered_biome_id(biome_params)
                        case 4 | 6:
                            return get_middle_biome_id(biome_params)
                case Continent.FarInland:
                    match biome_params.level_erosion:
                        case 0:
                            match biome_params.level_temperature:
                                case 0 | 1 | 2:
                                    if biome_params.weirdness < 0:
                                        return BiomeID.PeaksJagged
                                    elif biome_params.weirdness > 0:
                                        return BiomeID.PeaksFrozen
                                case 3:
                                    return BiomeID.PeaksStony
                                case 4:
                                    return get_badland_biome_id(biome_params)
                        case 1:
                            if biome_params.level_temperature < 3:
                                if biome_params.level_humidty < 2:
                                    return BiomeID.SnowySlope
                                else:
                                    return BiomeID.Grove
                            else:
                                return get_plateau_biome_id(biome_params)
                        case 2 | 3:
                            return get_plateau_biome_id(biome_params)
                        case 4 | 6:
                            return get_middle_biome_id(biome_params)
                        case 5:
                            return get_shattered_biome_id(biome_params)
        case PV.Peaks:
            match biome_params.continent:
                case Continent.Coast:
                    match biome_params.level_erosion:
                        case 0:
                            match biome_params.level_temperature:
                                case 0 | 1 | 2:
                                    if biome_params.weirdness < 0:
                                        return BiomeID.PeaksJagged
                                    elif biome_params.weirdness > 0:
                                        return BiomeID.PeaksFrozen
                                case 3:
                                    return BiomeID.PeaksStony
                                case 4:
                                    return get_badland_biome_id(biome_params)
                        case 1:
                            if biome_params.level_temperature == 0:
                                if biome_params.level_humidty < 2:
                                    return BiomeID.SnowySlope
                                else:
                                    return BiomeID.Grove
                            elif 0 < biome_params.level_temperature < 4:
                                return get_middle_biome_id(biome_params)
                            else:
                                return get_badland_biome_id(biome_params)
                        case 2 | 3 | 4:
                            return get_middle_biome_id(biome_params)
                        case 5:
                            if biome_params.weirdness < 0 or biome_params.level_temperature < 2 or biome_params.level_humidty == 4:
                                return get_shattered_biome_id(biome_params)
                            elif biome_params.weirdness > 0 and biome_params.level_temperature > 1 and biome_params.level_humidty < 4:
                                return BiomeID.SavannaWindsept
                        case 6:
                            return get_middle_biome_id(biome_params)
                case Continent.NearInland:
                    match biome_params.level_erosion:
                        case 0:
                            match biome_params.level_temperature:
                                case 0 | 1 | 2:
                                    if biome_params.weirdness < 0:
                                        return BiomeID.PeaksJagged
                                    elif biome_params.weirdness > 0:
                                        return BiomeID.PeaksFrozen
                                case 3:
                                    return BiomeID.PeaksStony
                                case 4:
                                    return get_badland_biome_id(biome_params)
                        case 1:
                            if biome_params.level_temperature == 0:
                                if biome_params.level_humidty < 2:
                                    return BiomeID.SnowySlope
                                else:
                                    return BiomeID.Grove
                            elif 0 < biome_params.level_temperature < 4:
                                return get_middle_biome_id(biome_params)
                            else:
                                return get_badland_biome_id(biome_params)
                        case 2 | 3 | 4:
                            return get_middle_biome_id(biome_params)
                        case 5:
                            if biome_params.weirdness < 0 or biome_params.level_temperature < 2 or biome_params.level_humidty == 4:
                                return get_shattered_biome_id(biome_params)
                            elif biome_params.weirdness > 0 and biome_params.level_temperature > 1 and biome_params.level_humidty < 4:
                                return BiomeID.SavannaWindsept
                        case 6:
                            return get_middle_biome_id(biome_params)
                case Continent.MidInland:
                    match biome_params.level_erosion:
                        case 0 | 1:
                            match biome_params.level_temperature:
                                case 0 | 1 | 2:
                                    if biome_params.weirdness < 0:
                                        return BiomeID.PeaksJagged
                                    elif biome_params.weirdness > 0:
                                        return BiomeID.PeaksFrozen
                                case 3:
                                    return BiomeID.PeaksStony
                                case 4:
                                    return get_badland_biome_id(biome_params)
                        case 2:
                            return get_plateau_biome_id(biome_params)
                        case 3:
                            if biome_params.level_temperature == 4:
                                return get_badland_biome_id(biome_params)
                            else:
                                return get_middle_biome_id(biome_params)
                        case 4 | 6:
                            return get_middle_biome_id(biome_params)
                        case 5:
                            return get_shattered_biome_id(biome_params)
                case Continent.FarInland:
                    match biome_params.level_erosion:
                        case 0 | 1:
                            match biome_params.level_temperature:
                                case 0 | 1 | 2:
                                    if biome_params.weirdness < 0:
                                        return BiomeID.PeaksJagged
                                    elif biome_params.weirdness > 0:
                                        return BiomeID.PeaksFrozen
                                case 3:
                                    return BiomeID.PeaksStony
                                case 4:
                                    return get_badland_biome_id(biome_params)
                        case 2 | 3:
                            return get_plateau_biome_id(biome_params)
                        case 4 | 6:
                            return get_middle_biome_id(biome_params)
                        case 5:
                            return get_shattered_biome_id(biome_params)
    return BiomeID.Null


def get_non_inland_biome_id(biome_params: BiomeParams) -> BiomeID:
    match biome_params.continent:
        case Continent.MushroomFields:
            return BiomeID.MushroomFields
        case Continent.Ocean:
            match biome_params.level_temperature:
                case 0:
                    return BiomeID.OceanDeepFrozen
                case 1:
                    return BiomeID.OceanDeepCold
                case 2:
                    return BiomeID.OceanDeep
                case 3:
                    return BiomeID.OceanDeepLukewarm
                case 4:
                    return BiomeID.OceanWarm
        case Continent.Ocean:
            match biome_params.level_temperature:
                case 0:
                    return BiomeID.OceanFrozen
                case 1:
                    return BiomeID.OceanCold
                case 2:
                    return BiomeID.Ocean
                case 3:
                    return BiomeID.OceanLukewarm
                case 4:
                    return BiomeID.OceanWarm
    return BiomeID.Null


def get_biome_id(biome_params: BiomeParams) -> BiomeID:
    if 0.2 <= biome_params.depth <= 0.9:
        if 0.8 <= biome_params.continentalness <= 1:
            return BiomeID.CavesDripstone
        if 0.7 <= biome_params.depth <= 1:
            return BiomeID.CavesLush
    elif biome_params.depth == 1.1:
        return BiomeID.DeepDark
    else:
        if biome_params.continent < Continent.Coast:
            return get_non_inland_biome_id(biome_params)
        return get_surface_biome_id(biome_params)


@dataclass
class Biome:
    _biome_params: BiomeParams

    def __post_init__(self):
        self.id: BiomeID = get_biome_id(self._biome_params)

    @property
    def temperature(self) -> float:
        return self._biome_params.temperature

    @property
    def humidity(self) -> float:
        return self._biome_params.humidity

    @property
    def continentalness(self) -> float:
        return self._biome_params.continentalness

    @property
    def weirdness(self) -> float:
        return self._biome_params.weirdness

    @property
    def value_peak_valley(self) -> float:
        return self._biome_params.value_peak_valley

    def to_nbt(self) -> nbtlib.Compound:
        return nbtlib.Compound({
            "Name": nbtlib.String(f"{self.id}")
        })
