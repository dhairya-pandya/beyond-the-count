import pandas as pd

from cpsa.biology.feature_families import aggregate_importance, parse_feature_name


def test_parse_feature_name_variants():
    assert parse_feature_name("Cells_AreaShape_Area") == ("Cells", "AreaShape", "none")
    assert parse_feature_name("Nuclei_Texture_Variance_RNA_5_01_256") == ("Nuclei", "Texture", "RNA")
    assert parse_feature_name("Cytoplasm_Intensity_IntegratedIntensity_Mito") == ("Cytoplasm", "Intensity", "Mito")
    assert parse_feature_name("Cells_Correlation_Overlap_AGP_DNA") == ("Cells", "Correlation", "AGP+DNA")
    assert parse_feature_name("Image_Granularity_10_ER") == ("Image", "Granularity", "ER")
    assert parse_feature_name("f_023") == ("unparsed", "unparsed", "unparsed")


def test_aggregate_importance_sums_to_one_per_endpoint():
    imp = pd.DataFrame({"Cells_AreaShape_Area": [0.2, 0.0], "Nuclei_Texture_Variance_RNA_5_01_256": [0.3, 0.5], "Image_Granularity_10_ER": [0.5, 0.5]},
                       index=["e1", "e2"])
    out = aggregate_importance(imp, level="group")
    assert out.sum(axis=1).round(6).tolist() == [1.0, 1.0]
    assert out.loc["e1", "AreaShape"] == 0.2 and out.loc["e2", "Texture"] == 0.5
