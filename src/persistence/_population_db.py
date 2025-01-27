import sqlite3
from numpy.typing import NDArray
from pymoo.core.individual import Individual


class PopulationDB:
    """Database class for population based experiments."""
    def __init__(self, path: str, num_parents: int=2) -> None:
        """
        Initialise the experiment database.

        :param path: Path to the database file.
        :param num_parents: Number of parents.
        """
        self.conn = sqlite3.connect(path)
        self.cursor = self.conn.cursor()
        self._num_parents = num_parents

        """Create table for indiviudals in the experiment."""
        parent_fields, parent_references = "", ""
        for i in range(num_parents):
            parent_fields += f"parent_{i} INTEGER,"
            parent_references += f"FOREIGN_KEY parent_{i} REFERENCES individuals (id),"

        self.cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS individuals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            generation INTEGER NOT NULL,
            fitness TEXT NOT NULL,
            {parent_fields}{parent_references})
            """
        )

        """Create table for genomes."""
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS genome (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            individual_id INTEGER NOT NULL,
            genome TEXT NOT NULL,
            FOREIGN KEY (individual_id) REFERENCES individuals (id))
            """
        )

        """Create table for expression of genomes."""
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS solution (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            individual_id INTEGER NOT NULL,
            solution TEXT NOT NULL,
            FOREIGN KEY (individual_id) REFERENCES individuals (id))
            """
        )

    def add_individual(
            self,
            generation: int,
            fitness: tuple[float, ...],
            genome: NDArray,
            solution: NDArray,
            parents: tuple[int, ...]
    ) -> None:
        """
        Add an individual to the database.

        :param generation: Current generation.
        :param fitness: Fitness of the individual.
        :param genome: Genome of the individual.
        :param solution: Expression of the genome.
        :param parents: The parents ids.
        """

        # Insert individual
        parent_cols = ",".join([f'parent_{i}' for i in range(self._num_parents)])
        qs = ",".join(["?",]*self._num_parents)
        self.cursor.execute(
            f"""INSERT INTO individuals (generation, fitness, {parent_cols}) VALUES (?, ?, {qs})""",
            (generation, fitness, *parents)
        )
        self.conn.commit()
        curr_id = self.cursor.lastrowid

        # Insert genome
        self.cursor.execute(
            """INSERT INTO genome (individual_id, genome) VALUES (?,?)""",
            (curr_id, genome)
        )
        self.conn.commit()

        # Insert solution
        self.cursor.execute(
            """INSERT INTO solution (individual_id, solution) VALUES (?, ?)""",
            (curr_id, solution)
        )
        self.conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()