import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from flightrecorder.cli import main
from flightrecorder.schema_registry import list_schema_records, load_schema, write_schema_bundle


class SchemaRegistryTests(unittest.TestCase):
    def test_catalog_loads_public_artifact_contracts(self):
        records = list_schema_records()
        names = {record["name"] for record in records}

        self.assertIn("scenario", names)
        self.assertIn("trace", names)
        self.assertIn("scorecard", names)
        self.assertIn("task_completion", names)
        self.assertIn("evidence_bundle", names)
        self.assertIn("training_manifest", names)
        self.assertIn("compare_rl_manifest", names)
        self.assertIn("review_manifest", names)
        self.assertIn("reviewed_manifest", names)
        for record in records:
            schema = load_schema(record["name"])
            self.assertEqual(schema["$id"], record["id"])
            self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
            self.assertEqual(schema["type"], "object")

    def test_load_schema_accepts_version_and_filename(self):
        by_name = load_schema("trace")
        by_version = load_schema("hfr.trace.v1")
        by_filename = load_schema("trace.v1.schema.json")

        self.assertEqual(by_name, by_version)
        self.assertEqual(by_name, by_filename)
        self.assertEqual(by_name["properties"]["schema_version"]["const"], "hfr.trace.v1")

    def test_write_schema_bundle_writes_catalog_and_selected_schemas(self):
        with tempfile.TemporaryDirectory() as tmp:
            written = write_schema_bundle(tmp, ["trace", "scorecard"])
            names = {path.name for path in written}

            self.assertEqual(names, {"manifest.json", "trace.v1.schema.json", "scorecard.v1.schema.json"})
            catalog = json.loads((Path(tmp) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual({record["name"] for record in catalog["schemas"]}, {"trace", "scorecard"})

    def test_cli_lists_and_exports_schemas(self):
        with tempfile.TemporaryDirectory() as tmp:
            stdout = StringIO()
            with redirect_stdout(stdout):
                list_code = main(["schemas"])
            self.assertEqual(list_code, 0)
            self.assertIn("trace\thfr.trace.v1\ttrace.v1.schema.json", stdout.getvalue())

            schema_out = Path(tmp) / "trace.schema.json"
            with redirect_stdout(StringIO()):
                export_code = main(["schemas", "--name", "trace", "--out", str(schema_out)])
            self.assertEqual(export_code, 0)
            exported = json.loads(schema_out.read_text(encoding="utf-8"))
            self.assertEqual(exported["properties"]["schema_version"]["const"], "hfr.trace.v1")

            bundle_dir = Path(tmp) / "bundle"
            with redirect_stdout(StringIO()):
                bundle_code = main(["schemas", "--name", "task_completion", "--write-dir", str(bundle_dir)])
            self.assertEqual(bundle_code, 0)
            self.assertTrue((bundle_dir / "manifest.json").exists())
            self.assertTrue((bundle_dir / "task_completion.v1.schema.json").exists())

            with self.assertRaises(SystemExit) as raised:
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    main(["schemas", "--name", "task_completion", "--write-dir", str(bundle_dir)])
            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
