import sqlite3
from numpy.typing import NDArray
from typing import Optional
import os.path


class PopulationDB:
    """Database class for population based experiments."""
    def __init__(self, path: str, num_parents: int=2, overwrite: bool = False) -> None:
        """
        Initialise the experiment database.

        :param path: Path to the database file.
        :param num_parents: Number of parents.
        :param overwrite: Whether to overwrite existing data.
        :raises FileExistsError: If file already exists.
        """
        if not overwrite and os.path.isfile(path):
            raise FileExistsError(f"File {path} already exists.")

        self.conn = sqlite3.connect(path)
        self.cursor = self.conn.cursor()
        self._num_parents = num_parents

        """Create table for indiviudals in the experiment."""
        parent_fields, parent_references = "", ""
        for i in range(num_parents):
            parent_fields += f"parent_{i} INTEGER, "
        self.cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS individuals (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            generation INTEGER NOT NULL, 
            fitness TEXT NOT NULL,
            {parent_fields[:-2]})
            """
        )
     

        """Create table for genomes."""
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS genome (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            individual_id INTEGER NOT NULL,
            genome TEXT NOT NULL)
            """
        )

        """Create table for expression of genomes."""
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS solution (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            individual_id INTEGER NOT NULL,
            solution TEXT NOT NULL)
            """
        )

    def add_individual(
            self,
            generation: int,
            fitness: tuple[float, ...],
            genome: NDArray,
            solution: NDArray,
            parents: Optional[tuple[int, ...]] = None
    ) -> None:
        """
        Add an individual to the database.

        :param generation: Current generation.
        :param fitness: Fitness of the individual.
        :param genome: Genome of the individual.
        :param solution: Expression of the genome.
        :param parents: The parents ids.
        """
        parents = [-1,] * self._num_parents if parents is None else parents

        # Insert individual
        parent_cols = ",".join([f'parent_{i}' for i in range(self._num_parents)])
        qs = ",".join(["?",]*self._num_parents)
        self.cursor.execute(
            f"""INSERT INTO individuals (generation, fitness, {parent_cols}) VALUES (?, ?, {qs})""",
            (generation, repr(fitness), *parents)
        )
        self.conn.commit()
        curr_id = self.cursor.lastrowid

        # Insert genome
        self.cursor.execute(
            """INSERT INTO genome (individual_id, genome) VALUES (?,?)""",
            (curr_id, repr(genome))
        )
        self.conn.commit()

        # Insert solution
        self.cursor.execute(
            """INSERT INTO solution (individual_id, solution) VALUES (?, ?)""",
            (curr_id, repr(solution))
        )
        self.conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()