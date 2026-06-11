#!/usr/bin/env python3
"""
ArcGIS World Imagery Satellite Image Fetcher
Downloads satellite imagery for properties using lat/long coordinates
"""

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FetchSettings:
    image_size: int = 256
    meters_per_pixel: float = 0.5
    output_dir: str = "satellite_images"
    request_delay: float = 0.1
    request_timeout: int = 30
    max_images: Optional[int] = None


class ArcGISImageFetcher:
    """Fetches satellite images from ArcGIS World Imagery service"""
    
    BASE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
    
    def __init__(self, settings: FetchSettings = FetchSettings()):
        """
        Initialize the image fetcher
        
        Args:
            settings: Runtime settings for image size, resolution, and output paths.
        """
        self.settings = settings
        self.image_size = settings.image_size
        self.meters_per_pixel = settings.meters_per_pixel
        self.output_dir = Path(settings.output_dir)
        self.session = requests.Session()
        
        # Create output directories
        self.train_dir = self.output_dir / "train"
        self.test_dir = self.output_dir / "test"
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized ArcGIS Image Fetcher")
        logger.info("Image size: %sx%s pixels", self.image_size, self.image_size)
        logger.info("Resolution: %sm per pixel", self.meters_per_pixel)
        logger.info("Output directory: %s", self.output_dir)

    @staticmethod
    def _load_table(path: str) -> pd.DataFrame:
        source_path = Path(path)
        if source_path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(source_path)
        if source_path.suffix.lower() == ".csv":
            return pd.read_csv(source_path)
        raise ValueError(f"Unsupported dataset format: {source_path.suffix}")

    @staticmethod
    def _has_valid_coordinates(lat: float, lon: float) -> bool:
        return np.isfinite(lat) and np.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180
    
    def lat_lon_to_bbox(self, lat: float, lon: float) -> Tuple[float, float, float, float]:
        """
        Convert lat/lon to bounding box for ArcGIS API
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Tuple of (xmin, ymin, xmax, ymax) in EPSG:4326
        """
        # Calculate the extent in degrees
        # Approximate: 1 degree latitude ≈ 111,320 meters
        # 1 degree longitude ≈ 111,320 * cos(latitude) meters
        
        extent_meters = (self.image_size / 2) * self.meters_per_pixel
        
        lat_degree_per_meter = 1.0 / 111320.0
        longitude_scale = max(abs(np.cos(np.radians(lat))), 1e-6)
        lon_degree_per_meter = 1.0 / (111320.0 * longitude_scale)
        
        delta_lat = extent_meters * lat_degree_per_meter
        delta_lon = extent_meters * lon_degree_per_meter
        
        xmin = lon - delta_lon
        ymin = lat - delta_lat
        xmax = lon + delta_lon
        ymax = lat + delta_lat
        
        return xmin, ymin, xmax, ymax
    
    def fetch_image(self, lat: float, lon: float, save_path: Path) -> bool:
        """
        Fetch a single satellite image
        
        Args:
            lat: Latitude
            lon: Longitude
            save_path: Path to save the image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self._has_valid_coordinates(lat, lon):
                logger.warning("Invalid coordinates skipped: lat=%s, lon=%s", lat, lon)
                return False

            xmin, ymin, xmax, ymax = self.lat_lon_to_bbox(lat, lon)
            
            # Construct API URL
            params = {
                "bbox": f"{xmin},{ymin},{xmax},{ymax}",
                "bboxSR": "4326",  # WGS84 coordinate system
                "size": f"{self.image_size},{self.image_size}",
                "format": "png",
                "f": "image"
            }
            
            response = self.session.get(
                self.BASE_URL,
                params=params,
                timeout=self.settings.request_timeout
            )
            response.raise_for_status()
            
            with open(save_path, "wb") as f:
                f.write(response.content)
            
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error("Failed to fetch image for lat=%s, lon=%s: %s", lat, lon, e)
            return False
        except Exception as e:
            logger.error("Unexpected error for lat=%s, lon=%s: %s", lat, lon, e)
            return False
    
    def fetch_dataset_images(self, 
                            csv_path: str, 
                            dataset_type: str = 'train',
                            delay: Optional[float] = None,
                            max_images: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch images for entire dataset
        
        Args:
            csv_path: Path to Excel file with lat/long data
            dataset_type: 'train' or 'test'
            delay: Delay between requests in seconds (to be respectful)
            max_images: Maximum number of images to download (for testing)
            
        Returns:
            DataFrame with added 'image_path' column
        """
        logger.info("Loading dataset from %s", csv_path)
        df = self._load_table(csv_path)
        
        logger.info("Dataset shape: %s", df.shape)
        logger.info("Columns: %s", df.columns.tolist())

        required_cols = {"id", "lat", "long"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing_cols)}")
        
        # Determine output directory
        output_dir = self.train_dir if dataset_type == "train" else self.test_dir
        delay = self.settings.request_delay if delay is None else delay
        max_images = self.settings.max_images if max_images is None else max_images
        
        # Limit dataset if specified
        if max_images:
            df = df.head(max_images)
            logger.info("Limited to %s images for testing", max_images)
        
        # Add image path column
        df["image_path"] = ""
        
        success_count = 0
        fail_count = 0
        
        logger.info("Starting to download %s images...", len(df))
        
        for idx, row in df.iterrows():
            property_id = row["id"]
            lat = row["lat"]
            lon = row["long"]
            
            # Create filename using property ID
            filename = f"{property_id}.png"
            save_path = output_dir / filename
            
            # Skip if already exists
            if save_path.exists():
                df.at[idx, "image_path"] = str(save_path)
                logger.info("[%s/%s] Skipped existing image: %s", idx + 1, len(df), filename)
                success_count += 1
                continue
            
            # Fetch image
            if self.fetch_image(lat, lon, save_path):
                df.at[idx, "image_path"] = str(save_path)
                success_count += 1
                logger.info("[%s/%s] Downloaded: %s", idx + 1, len(df), filename)
            else:
                fail_count += 1
                logger.warning("[%s/%s] Failed: %s", idx + 1, len(df), filename)
            
            # Respectful delay between requests
            time.sleep(delay)
            
            # Progress update every 100 images
            if (idx + 1) % 100 == 0:
                logger.info(
                    "Progress: %s/%s | Success: %s | Failed: %s",
                    idx + 1,
                    len(df),
                    success_count,
                    fail_count
                )
        
        logger.info("Download complete!")
        logger.info("Total: %s | Success: %s | Failed: %s", len(df), success_count, fail_count)
        
        # Save updated dataframe with image paths
        output_csv = self.output_dir / f"{dataset_type}_with_images.csv"
        df.to_csv(output_csv, index=False)
        logger.info("Saved dataset with image paths to: %s", output_csv)
        
        return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ArcGIS satellite images for train/test housing datasets."
    )
    parser.add_argument("--train-file", default="datasets/train(1).xlsx")
    parser.add_argument("--test-file", default="datasets/test2.xlsx")
    parser.add_argument("--output-dir", default="satellite_images")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--meters-per-pixel", type=float, default=0.5)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("ArcGIS World Imagery Satellite Image Fetcher")
    logger.info("=" * 60)
    
    settings = FetchSettings(
        image_size=args.image_size,
        meters_per_pixel=args.meters_per_pixel,
        output_dir=args.output_dir,
        request_delay=args.delay,
        max_images=args.max_images
    )

    fetcher = ArcGISImageFetcher(settings)
    
    # Download training images
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Downloading TRAINING images")
    logger.info("=" * 60)
    train_df = fetcher.fetch_dataset_images(
        csv_path=args.train_file,
        dataset_type="train"
    )
    
    # Download test images
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Downloading TEST images")
    logger.info("=" * 60)
    test_df = fetcher.fetch_dataset_images(
        csv_path=args.test_file,
        dataset_type="test"
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("ALL DONE!")
    logger.info("=" * 60)
    logger.info("Training images: %s", len(train_df))
    logger.info("Test images: %s", len(test_df))
    logger.info("Images saved in: %s", fetcher.output_dir)


if __name__ == "__main__":
    main()