# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import datetime
import os
import shutil
import json
from pathlib import Path

from ams.util import get_unique_fn
from ams.util import mkdir
from ams.store_types import AMSModelDescr
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine, Column, Integer, String, Enum, Text, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
import enum
from typing import List
from collections import defaultdict


Base = declarative_base()


class EntryType(enum.Enum):
    candidates = "candidates"
    models = "models"
    data = "data"


class Entry(Base):
    __tablename__ = "entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    application_name = Column(String(255), nullable=False)
    domain_name = Column(String(255), nullable=False)
    filename = Column(Text, nullable=False, unique=True)
    entry_type = Column(Enum(EntryType), nullable=False)
    version = Column(Integer, nullable=False, default=1)
    meta_dict = Column("metadata", JSON, nullable=True)


class AMSDataStore:
    """A class representing the persistent data of AMS.

    The class abstracts the 'view' of AMS persistent data storage through
    a SQL database that stores information reqarding different files in the Fileystem.

    The SQL database catecorizes files in three possible types:
        1. 'data' : A collection of files stored in some PFS directory that will train some model
        2. 'models' : A collection of torch-scripted models.
        3. 'candidates' : A collection of files stored in some PFS directory that can be added as data.

    Every 'entry' is associated with a domainName. Providing the persistent abstraction.
    'EOS maps to a set of files and models'
    """

    entry_suffix = {"data": "h5", "models": "pt", "candidates": "h5"}
    entry_mime_types = {"data": "hdf5", "models": "zip", "candidates": "hdf5"}
    valid_entries = {"data", "candidates", "models"}
    valid_dbs = {"sqlite", "mariadb"}

    def __init__(self, application_name, url):
        """
        Initializes the AMSDataStore class. Upon init the kosh-store is closed and not connected
        """
        self._application_name = application_name
        self._url = url
        try:
            print(f"DB url is {self._url.split('@')[1]}")
        except IndexError as _:
            print(f"Error: url seems to be malformed (missing @)")
        self._session = None
        self._engine = None

    def is_open(self):
        """
        Check whether we are connected to a database
        """

        return self._session is not None

    def open(self):
        """
        Open and connect to the database
        """
        if self.is_open():
            return self

        self._engine = create_engine(self._url)
        Base.metadata.create_all(self._engine)
        self._session = sessionmaker(bind=self._engine)

        return self

    def close(self):
        if self._engine:
            self._engine.dispose()
        self._engine = None
        self._session = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def find(self, domain_name=None, filename=None, entry_type=None, version=None):
        session = self._session()
        try:
            query = session.query(Entry)
            query = query.filter(Entry.application_name == self._application_name)
            if domain_name is not None:
                query = query.filter(Entry.domain_name == domain_name)
            if filename is not None:
                query = query.filter(Entry.filename == filename)
            if entry_type is not None:
                if isinstance(entry_type, str):
                    entry_type = EntryType(entry_type)
                query = query.filter(Entry.entry_type == entry_type)
            if version is not None:
                query = query.filter(Entry.version == version)

            return query.all()
        finally:
            session.close()

    def _add_entries(self, domain_name: str, entry_type: str, filenames: List[str], version=None, metadata=None):
        """
        Adds files of entry_type on the designated domain_name and associates the version and the metadata to those entries.
        Args:
            domain_name: The domain_name of this entry
            entry_type: Can be either 'models', 'candidates', 'data'.
            filenames: A list of files to add in the entry
            version: The version to assign to all files
            meta_dict: The metadata to associate with this file

        Returns:

            None
        """
        if entry_type not in AMSDataStore.valid_entries:
            raise ValueError(f"{add_entries} Expets a 'entry_type' to be in {AMSDataStore.valid_entries}")

        abs_path = []
        for fn in filenames:
            if not Path(fn).exists():
                raise RuntimeError("Adding a non existend file to store is not supported")
            abs_path.append(str(Path(fn).resolve()))

        session = self._session()
        try:
            if isinstance(entry_type, str):
                entry_type = EntryType(entry_type)

            if version is None:
                max_version = (
                    session.query(func.max(Entry.version))
                    .filter_by(application_name=self._application_name, domain_name=domain_name, entry_type=entry_type)
                    .scalar()
                )

                version = (max_version or 0) + 1

            entries = [
                Entry(
                    application_name=self._application_name,
                    domain_name=domain_name,
                    filename=fn,
                    entry_type=entry_type,
                    version=version,
                    meta_dict=metadata,
                )
                for fn in abs_path
            ]

            session.add_all(entries)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def add_data(self, domain_name, data_files=list(), version=None, metadata=dict()):
        """
        Adds files in the kosh-store and associates them to the 'data' entry.

        The function assumes data_files to always be in hdf5 format.

        Args:
            data_files: A list of files to add in the entry
            version: The version to assign to all files
            metadata: The metadata to associate with this file
        """
        self._add_entries(domain_name, entry_type="data", filenames=data_files, version=version, metadata=metadata)

    def add_model(self, domain_name, model, test_error, val_error, version=None, metadata=dict()):
        """
        Adds a model in the kosh-store and associates them to the 'models' entry.

        The function assumes models to always be in torchscript format.

        Args:
            model: The path containing the torchscript model
            version: The version to assing to the model
            metadata: The metadata to associate with this model
        """
        if not isinstance(model, AMSModelDescr):
            raise TypeError(f"AMSStore expects AMSModelDescr as a model-entry, got {type(model)}")

        info = model.to_dict()
        del info["path"]
        info["val_error"] = val_error
        info["test_error"] = test_error
        for k, v in metadata.items():
            if k in info.keys():
                raise RuntimeError(f"Key {k} exists in both info and metadata")
            else:
                info[k] = str(v)
        self._add_entries(domain_name, entry_type="models", filenames=[model.path], version=version, metadata=info)

    def add_candidates(self, domain_name, data_files=list(), version=None, metadata=dict()):
        """
        Adds files in the kosh-store and associates them to the 'candidates' entry.

        The function assumes candidates to always be in hdf5 format.

        Args:
            data_files: A list of candidate files
            version: The version to assign to the model
            metadata: The metadata to associate with this model
        """
        self._add_entries(
            domain_name, entry_type="candidates", filenames=data_files, version=version, metadata=metadata
        )

    def _remove_entries(
        self, domain_name: str, entry_type: str, filenames: List[str], version=None, metadata=None, purge=True
    ):
        """
        Remove files from database and from filesystem

        Args:
            domain_name: The domain name this files belong to
            entry_type: The entry to look for the specified files
            filenames: A list of files to be deleted
            version: An integer or none if we need to filter based on version
            metadata= Additional metadata we can query and partially delete from
        """

        abs_path = []
        for fn in filenames:
            if not Path(fn).exists():
                raise RuntimeError("Deleting a non existend file to store is not supported")
            abs_path.append(str(Path(fn).resolve()))

        session = self._session()
        try:
            if isinstance(entry_type, str):
                entry_type = EntryType(entry_type)

            # Step 1: Query matching entries
            query = session.query(Entry).filter(
                Entry.application_name == self._application_name,
                Entry.domain_name == domain_name,
                Entry.entry_type == entry_type,
                Entry.filename.in_(abs_path),
            )

            if metadata:
                for key, value in metadata.items():
                    query = query.filter(Entry.meta_dict[key].astext == str(value))

            entries_to_delete = query.all()
            if not entries_to_delete:
                print("No matching entries found.")
                return

            # Step 2: Get list of filenames to delete
            files_to_delete = [entry.filename for entry in entries_to_delete]

            # Step 3: Delete DB entries
            for entry in entries_to_delete:
                session.delete(entry)

            session.commit()
            if purge:
                for filepath in files_to_delete:
                    try:
                        fn = Path(filepath)
                        if fn.exists():
                            fn.unlink()
                        else:
                            print(f"File not found: {filepath}")
                    except Exception as file_err:
                        print(f"Error deleting file {filepath}: {file_err}")

        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def remove_data(self, domain_name: str, filenames: List[str], version=None, metadata=None, purge=True):
        """
        Remove files from the data database and from filesystem

        Args:
            domain_name: The domain name this files belong to
            entry_type: The entry to look for the specified files
            filenames: A list of files to be deleted
            version: An integer or none if we need to filter based on version
            metadata: Additional metadata we can query and partially delete from
            purge: If set to true it will delete the file from the filesystem
        """

        self._remove_entries(domain_name, "data", filenames, version, metadata, purge)

    def remove_models(self, domain_name: str, models: List[str], version=None, metadata=None, purge=True):
        """
        Remove files from the data database and from filesystem

        Args:
            domain_name: The domain name this files belong to
            entry_type: The entry to look for the specified files
            models: A list of files to be deleted
            version: An integer or none if we need to filter based on version
            metadata: Additional metadata we can query and partially delete from
            purge: If set to true it will delete the file from the filesystem
        """
        self._remove_entries(domain_name, "models", models, version, metadata, purge)

    def remove_candidates(self, domain_name: str, filenames: List[str], version=None, metadata=None, purge=True):
        """
        Remove files from the data database and from filesystem

        Args:
            domain_name: The domain name this files belong to
            entry_type: The entry to look for the specified files
            filenames: A list of files to be deleted
            version: An integer or none if we need to filter based on version
            metadata: Additional metadata we can query and partially delete from
            purge: If set to true it will delete the file from the filesystem
        """

        self._remove_entries(domain_name, "candidates", filenames, version, metadata, purge)

    def _get_entry_versions(self, domain_name, entry_type, associate_files=False):
        """
        Returns a list of versions existing for the specified entry

        Args:
            domain_name: The entry type we are looking for
            associate_files: Associate files in store with the versions

        Returns:
            A list of the unique existing versions in our database or a dictionary of versions to lists associating files with the specific version
        """

        session = self._session()
        try:
            if isinstance(entry_type, str):
                entry_type = EntryType(entry_type)

            query = session.query(Entry).filter(Entry.entry_type == entry_type)
            query = query.filter(Entry.application_name == self._application_name)
            query = session.query(Entry).filter(Entry.entry_type == entry_type)
            query = query.filter(Entry.domain_name == domain_name)

            entries = query.all()

            if associate_files:
                result = defaultdict(list)
                for entry in entries:
                    result[entry.version].append(entry.filename)

                return dict(result)
            else:
                result = []
                for entry in entries:
                    result.append(entry.version)
                return list(set(result))
        finally:
            session.close()

    def get_data_versions(self, domain_name, associate_files=False):
        """
        Returns a list of versions existing for the data entry


        Returns:
            A list of existing versions in our database
        """
        return self._get_entry_versions(domain_name, "data", associate_files)

    def get_model_versions(self, domain_name, associate_files=False):
        """
        Returns a list of versions existing for the model entry


        Returns:
            A list of existing model versions in our database
        """

        return self._get_entry_versions(domain_name, "models", associate_files)

    def get_candidate_versions(self, domain_name, associate_files=False):
        """
        Returns a list of versions existing for the candidate entry


        Returns:
            A list of existing candidate versions in our database
        """

        return self._get_entry_versions(domain_name, "candidates", associate_files)

    def get_files(self, domain_name: str, entry: str, versions=None):
        """
        Returns a list of paths to files for the specified version

        Args:
            entry: The entry in the ensemble can be any of candidates, model, data
            versions: A list of versions we are looking for.
                If 'None'   return all files in entry
                If "latest" return the latest version in the store

        Returns:
            A list of existing files in the kosh-store
        """
        files = self._get_entry_versions(domain_name, entry, True)

        if len(files) == 0:
            return list()

        if isinstance(versions, str) and versions == "latest":
            max_version = max(files.keys())
            return files[max_version]

        file_paths = list()
        for k, v in files.items():
            if versions is None or v in versions:
                file_paths = file_paths + v
        return file_paths

    def get_candidate_files(self, domain_name, versions=None):
        """
        Returns a list of paths to files for the specified version

        Args:
            versions: A list of versions we are looking for.
                If 'None'   return all files in entry
                If "latest" return the latest version in the store
                If "list" return only files matching these versions

        Returns:
            A list of existing files in the kosh-store candidates ensemble
        """
        return self.get_files(domain_name, "candidates", versions)

    def get_model_files(self, domain_name, versions=None):
        """
        Returns a list of paths to files for the specified version

        Args:
            versions: A list of versions we are looking for.
                If 'None'   return all files in entry
                If "latest" return the latest version in the store
                If "list" return only files matching these versions

        Returns:
            A list of existing files in the kosh-store model ensemble
        """

        return self.get_files(domain_name, "models", versions)

    def get_data_files(self, domain_name, versions=None):
        """
        Returns a list of paths to files for the specified version

        Args:
            versions: A list of versions we are looking for.
                If 'None'   return all files in entry
                If "latest" return the latest version in the store
                If "list" return only files matching these versions

        Returns:
            A list of existing files in the kosh-store model ensemble
        """

        return self.get_files(domain_name, "data", versions)

    def move(self, domain_name, dest_root_path, src_type, dest_type, filenames):
        """
        Moves files between direcories and updates the respective db. It follows a "safe" approach: copy, add, delete the file instead of moving the underlying file.

        Args:
            src_type: the ensemble name containing the original files
            dst_root_path: The directory to which we should move directories to
            dest_type: the ensemble name of the files
            files: The files to be moved

        NOTE: The current implementation will lose all metadata associated with the original src files. We need to consider whether we want to "migrate"
        those to the destination entry dataset.
        """

        session = self._session()
        try:
            if isinstance(src_type, str):
                src_type = EntryType(src_type)
            if isinstance(dest_type, str):
                dest_type = EntryType(dest_type)

            # 1. Query all matching entries
            entries = (
                session.query(Entry)
                .filter(
                    Entry.application_name == self._application_name,
                    Entry.domain_name == domain_name,
                    Entry.entry_type == src_type,
                    Entry.filename.in_(filenames),
                )
                .all()
            )

            if not entries:
                print("No matching entries found to promote.")
                return

            # 2. Prepare move plan
            move_plan = []
            for entry in entries:
                src_path = Path(entry.filename)
                dest_path = Path(dest_root_path) / src_path.name
                move_plan.append((entry, src_path, dest_path))

            # 3. Copy files to new location first
            for _, src_path, dest_path in move_plan:
                if not src_path.exists():
                    raise FileNotFoundError(f"Source file not found: {src_path}")
                shutil.copy2(src_path, dest_path)

            # 4. Update DB entries
            for entry, _, dst_path in move_plan:
                entry.filename = str(dst_path.resolve())
                entry.entry_type = dest_type

            session.commit()
            print(f"Database updated. Promoted {len(entries)} entries to '{dest_type.value}'.")

            # 5. After successful DB update, delete originals
            for _, src_path, _ in move_plan:
                try:
                    src_path.unlink()
                    print(f"Deleted original file: {src_path}")
                except Exception as cleanup_err:
                    print(f"Could not delete original file: {src_path}: {cleanup_err}")

        except (SQLAlchemyError, OSError, FileNotFoundError) as e:
            session.rollback()
            print(f"Move failed: {e}")
            raise e
        finally:
            session.close()

    def search(self, domain_name=None, entry=None, version=None, metadata=dict()):
        """
        Search for items in the database that match the metadata
        Args:
            entry: Which entry to search for ('data', 'models', 'candidates')
            version: Specific version to look for, when 'version' is 'latest' we
                return the entry with the largest version. If None, we are not matching
                versions.
            metadata: A dictionary of key values to search in our database

        Returns:
            A list of matching entries described as dictionaries
        """
        latest = False
        if (version is not None) and (version == "latest"):
            latest = True
            version = None

        entries = self.find(domain_name, entry_type=entry, version=version)

        result = []

        for entry in entries:
            result.append(
                {
                    "id": entry.id,
                    "application_name": entry.application_name,
                    "domain_name": entry.domain_name,
                    "filename": entry.filename,
                    "entry_type": entry.entry_type.value,
                    "version": entry.version,
                    "metadata": entry.meta_dict,
                }
            )

        if len(result) != 0 and latest:
            result = [max(result, key=lambda item: item["version"])]

        return result

    def __str__(self):
        return "AMS Store(name={0}, status={2})".format(
            self._application_name, "Open" if self.is_open() else "Closed"
        )

    def _suggest_entry_file_name(self, entry, domain_name):
        if domain_name is None:
            return str(Path(f"{get_unique_fn()}.{self.__class__.entry_suffix[entry]}"))
        return str(
            Path(
                f"{domain_name}_{get_unique_fn()}.{self.__class__.entry_suffix[entry]}"
            )
        )

    def suggest_model_file_name(self, domain_name=None):
        return self._suggest_entry_file_name("models", domain_name)

    def suggest_candidate_file_name(self, domain_name=None):
        return self._suggest_entry_file_name("candidates", domain_name)

    def suggest_data_file_name(self, domain_name=None):
        return self._suggest_entry_file_name("data", domain_name)


def create_store_directories(store_path):
    """
    Creates the directory structure AMS prefers under the store_path.
    """
    store_path = Path(store_path)
    if not store_path.exists():
        store_path.mkdir(parents=True, exist_ok=True)

    for fn in list(AMSDataStore.valid_entries):
        mkdir(store_path, fn)
