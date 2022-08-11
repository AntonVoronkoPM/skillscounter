from datetime import datetime

from feast import Entity, Feature, FeatureView, ValueType
from feast.data_source import FileSource
from google.protobuf.duration_pb2 import Duration

# Read data
START_TIME = "2022-02-17"
project_details = FileSource(
    path="../data/full_dataset.parquet",
    event_timestamp_column="created_on",
)

# Define an entity for the project
project = Entity(
    name="id",
    value_type=ValueType.INT64,
    description="item id",
)

# Define a Feature View for each project
# Can be used for fetching historical data and online serving
project_details_view = FeatureView(
    name="item_details",
    entities=["id"],
    ttl=Duration(
        seconds=(datetime.today() - datetime.strptime(START_TIME, "%Y-%m-%d")).days * 24 * 60 * 60
    ),
    features=[
        Feature(name="text", dtype=ValueType.STRING),
        Feature(name="tag", dtype=ValueType.STRING),
        Feature(name="target", dtype=ValueType.STRING),
    ],
    online=True,
    input=project_details,
    tags={},
)
