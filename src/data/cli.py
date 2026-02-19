"""Command-line interface for mining segmentation downloader."""

import argparse
import logging
from pathlib import Path

from .database import DownloadDatabase
from .config import Config

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Setup logging configuration.
    
    Args:
        verbose: Enable debug logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Silence rasterio logging
    logging.getLogger('rasterio').setLevel(logging.ERROR)
    logging.getLogger('rasterio.env').setLevel(logging.ERROR)
    logging.getLogger('rasterio._env').setLevel(logging.ERROR)
    logging.getLogger('rasterio._base').setLevel(logging.ERROR)
    logging.getLogger('zarr.group').setLevel(logging.ERROR)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Download Landsat data for mining regions'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Create tasks command
    create_parser = subparsers.add_parser('create', help='Create download tasks')
    create_parser.add_argument(
        '--mining-file',
        type=str,
        required=True,
        help='Path to mining polygons GeoPackage file'
    )
    create_parser.add_argument(
        '--countries',
        type=str,
        nargs='+',
        help='ISO3 country codes (e.g., ZAF USA). If not specified, all countries.'
    )
    create_parser.add_argument(
        '--buffer',
        type=float,
        default=0.01,
        help='Buffer size in degrees around each mine for clustering (default: 0.01)'
    )
    
    # Run workers command
    run_parser = subparsers.add_parser('run', help='Run workers')
    run_parser.add_argument(
        '--workers',
        type=str,
        nargs='+',
        choices=['export', 'status', 'download', 'janitor', 'tasks', 'all'],
        default=['all'],
        help='Workers to run (default: all)'
    )
    run_parser.add_argument(
        '--countries',
        type=str,
        nargs='+',
        help='ISO3 country codes to filter tasks (e.g., ZAF USA). If not specified, all countries.'
    )
    run_parser.add_argument(
        '--once',
        action='store_true',
        help='Run once instead of continuously'
    )
    run_parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        help='Years to generate tasks for (e.g., 2020 2021 2022). If not specified, 1984-2023.'
    )

    # One-time store command: convert downloaded TIFFs to Zarr
    store_parser = subparsers.add_parser('store', help='Convert downloaded TIFFs to Zarr (one-time)')
    store_parser.add_argument(
        '--countries',
        type=str,
        nargs='+',
        help='ISO3 country codes to filter tasks (e.g., ZAF USA). If not specified, all countries.'
    )
    store_parser.add_argument(
        '--input-dir',
        type=str,
        default=None,
        help='Directory containing downloaded .tif and .json files (overrides DB-driven listing)'
    )
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show download status')
    
    # Transfer zarr command
    transfer_zarr_parser = subparsers.add_parser('transfer-zarr', help='Transfer zarr data to HPC')

    # Transfer downloads command (new)
    transfer_downloads_parser = subparsers.add_parser('transfer-downloads', help='Transfer downloaded TIFFs to HPC')
    transfer_downloads_parser.add_argument('--limit', type=int, default=None, help='Limit number of files to transfer')
    transfer_downloads_parser.add_argument('--batch-size', type=int, default=None, help='Files per archive (overrides config)')
    transfer_downloads_parser.add_argument('--scan-dir', action='store_true', help='Scan DOWNLOAD_DIR instead of DB for files')
    transfer_downloads_parser.add_argument('--dry-run', action='store_true', help='Show what would be transferred without performing the transfer')

    # Common arguments
    for p in [create_parser, run_parser, store_parser, status_parser, transfer_zarr_parser, transfer_downloads_parser]:
        p.add_argument(
            '--db',
            type=str,
            default=None,
            help='Database path (default: from config)'
        )
        p.add_argument(
            '--verbose',
            '-v',
            action='store_true',
            help='Enable verbose logging'
        )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    setup_logging(args.verbose)
    
    # Initialize config and database
    config = Config()
    config.ensure_dirs()
    
    db_path = args.db if hasattr(args, 'db') and args.db else config.DB_PATH
    db = DownloadDatabase(str(db_path))
    
    # Execute command
    if args.command == 'create':
        from .clustering import create_clusters_and_tiles
        create_clusters_and_tiles(
            args.mining_file,
            args.countries,
            args.buffer,
            db
        )
    
    elif args.command == 'run':
        from .gee_export import GEEExportWorker
        from .status_checker import StatusCheckerWorker
        from .download import DownloadWorker
        from .tasks import TaskGeneratorWorker
        from .janitor import JanitorWorker
        import threading
        
        continuous = not args.once
        workers_to_run = args.workers
        countries = args.countries if hasattr(args, 'countries') else None
        years = args.years if hasattr(args, 'years') and args.years else None
        
        if 'all' in workers_to_run:
            workers_to_run = ['tasks', 'export', 'status', 'download', 'janitor']
        
        if countries:
            logger.info(f"Filtering tasks for countries: {', '.join(countries)}")
        
        threads = []
        
        if 'tasks' in workers_to_run:
            logger.info("Starting task generator worker")
            worker = TaskGeneratorWorker(db, config, countries=countries, years=years)
            thread = threading.Thread(target=worker.run, args=(continuous,))
            thread.start()
            threads.append(thread)
        
        if 'export' in workers_to_run:
            logger.info("Starting export worker")
            worker = GEEExportWorker(db, config, countries=countries)
            thread = threading.Thread(target=worker.run, args=(continuous,))
            thread.start()
            threads.append(thread)
        
        if 'status' in workers_to_run:
            logger.info("Starting status checker worker")
            worker = StatusCheckerWorker(db, config, countries=countries)
            thread = threading.Thread(target=worker.run, args=(continuous,))
            thread.start()
            threads.append(thread)
        
        if 'download' in workers_to_run:
            logger.info("Starting download worker")
            worker = DownloadWorker(db, config, countries=countries)
            thread = threading.Thread(target=worker.run, args=(continuous,))
            thread.start()
            threads.append(thread)
        
        if 'janitor' in workers_to_run:
            logger.info("Starting janitor worker")
            from .janitor import JanitorWorker
            worker = JanitorWorker(db, config, countries=countries, clean=True)
            thread = threading.Thread(target=worker.run, args=(continuous,))
            thread.start()
            threads.append(thread)
        
        # Wait for all threads
        for thread in threads:
            thread.join()
    
    elif args.command == 'status':
        stats = db.get_statistics()
        
        print("\n=== Download Statistics ===")
        print(f"\nTotal tasks: {stats['total_tasks']}")
        
        print("\nStatus breakdown:")
        for status, count in stats['status_counts'].items():
            print(f"  {status}: {count}")
        
        print("\nCountries:")
        for country, count in stats['country_counts'].items():
            print(f"  {country}: {count} polygons")
        
        if stats['year_range']['min_year']:
            print(f"\nYear range: {stats['year_range']['min_year']} - {stats['year_range']['max_year']}")
    
    elif args.command == 'store':
        # One-time conversion of downloaded TIFFs into the Zarr store
        from .store import process_downloaded_tasks

        countries = args.countries if hasattr(args, 'countries') else None
        input_dir = args.input_dir if hasattr(args, 'input_dir') and args.input_dir else None
        logger.info("Starting one-time storage (convert TIFFs → Zarr)")
        processed = process_downloaded_tasks(config=config, countries=countries, input_dir=input_dir)
        logger.info(f"One-time store completed: processed {processed} tasks")

    elif args.command == 'transfer-zarr':
        # Legacy alias — delegates to the transfer-downloads implementation.
        from .transfer_downloads import find_files_from_db, transfer_batches
        logger.info("Starting downloads transfer to HPC (transfer-zarr alias)")
        files = find_files_from_db(db)
        transferred = transfer_batches(files, config)
        logger.info(f"Downloads transfer completed: {transferred} files")

    elif args.command == 'transfer-downloads':
        from .transfer_downloads import find_files_from_db, find_files_by_scan, transfer_batches
        logger.info('Starting downloads transfer to HPC')
        if getattr(args, 'scan_dir', False):
            logger.info(f"Scanning download directory: {config.DOWNLOAD_DIR}")
            files = find_files_by_scan(config.DOWNLOAD_DIR, limit=getattr(args, 'limit', None))
        else:
            files = find_files_from_db(db, limit=getattr(args, 'limit', None))
        transferred = transfer_batches(
            files, config,
            batch_size=getattr(args, 'batch_size', None),
            dry_run=getattr(args, 'dry_run', False),
        )
        logger.info(f'Downloads transfer completed: {transferred} files')



if __name__ == '__main__':
    main()