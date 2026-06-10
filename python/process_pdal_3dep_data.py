import pdal
import json

pipeline_def = {
    "pipeline": [
        {
            "type": "readers.ept",
            "filename": "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/VA_NShenandoah_1_2020/ept.json",
            "bounds": "([-8794240, -8709637], [4545577, 4621893] )",
            "resolution": 1.0
        },
        {
            "type": "filters.range",
            "limits": "Classification[2:2]"   # ground points only
        },
        {
            "type": "writers.las",
            "filename": "albemarle_2020_ground.laz",
            "compression": "laszip"
        }
    ]
}

pipeline = pdal.Pipeline(json.dumps(pipeline_def))
count = pipeline.execute()
print(f"Ground points extracted: {count:,}")

