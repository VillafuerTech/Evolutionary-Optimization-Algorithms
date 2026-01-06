"""Class scheduling optimization task."""

from typing import Any


# Default demo data
DEFAULT_COURSES = [
    {"name": "Data Structures", "professor": "Prof. Smith", "students": 25},
    {"name": "Programming 2", "professor": "Prof. Johnson", "students": 25},
    {"name": "Discrete Math", "professor": "Prof. Williams", "students": 25},
    {"name": "C++ Programming", "professor": "Prof. Brown", "students": 25},
    {"name": "Databases", "professor": "Prof. Smith", "students": 25},
    {"name": "Networks", "professor": "Prof. Davis", "students": 25},
    {"name": "Advanced C++", "professor": "Prof. Johnson", "students": 25},
    {"name": "Cybersecurity", "professor": "Prof. Miller", "students": 25},
    {"name": "Operating Systems", "professor": "Prof. Wilson", "students": 25},
    {"name": "AI", "professor": "Prof. Wilson", "students": 25},
    {"name": "Machine Learning", "professor": "Prof. Taylor", "students": 25},
    {"name": "Computer Org", "professor": "Prof. Taylor", "students": 25},
]

DEFAULT_PROFESSOR_AVAILABILITY = {
    "Prof. Smith": [True, True, False, False, False, True, True],
    "Prof. Johnson": [False, True, True, False, True, False, False],
    "Prof. Williams": [True, False, False, True, True, False, True],
    "Prof. Brown": [True, True, True, False, False, True, False],
    "Prof. Davis": [False, False, True, True, False, True, False],
    "Prof. Miller": [True, False, True, True, False, True, False],
    "Prof. Wilson": [True, True, False, False, True, False, True],
    "Prof. Taylor": [False, True, False, True, False, True, True],
}

DEFAULT_CLASSROOMS = [
    {"name": "Room 101", "capacity": 25},
    {"name": "Room 102", "capacity": 25},
    {"name": "Room 103", "capacity": 25},
    {"name": "Room 104", "capacity": 25},
    {"name": "Room 105", "capacity": 25},
]

DEFAULT_TIME_SLOTS = [
    "08:30-10:00",
    "10:00-11:30",
    "11:30-13:00",
    "13:00-14:30",
    "14:30-16:00",
    "16:00-17:30",
    "17:30-19:00",
]


class SchedulingTask:
    """Class scheduling optimization task.

    Optimizes assignment of courses to time slots and classrooms
    while respecting professor availability and classroom capacity.
    """

    task_type = "scheduling"
    optimization_type = "min"
    name = "scheduling"

    def __init__(
        self,
        courses: list[dict[str, Any]] | None = None,
        professor_availability: dict[str, list[bool]] | None = None,
        classrooms: list[dict[str, Any]] | None = None,
        time_slots: list[str] | None = None,
    ):
        """Initialize scheduling task.

        Args:
            courses: List of course dicts with name, professor, students.
            professor_availability: Dict mapping professor to availability list.
            classrooms: List of classroom dicts with name, capacity.
            time_slots: List of time slot strings.
        """
        self.courses = courses or DEFAULT_COURSES
        self.professor_availability = professor_availability or DEFAULT_PROFESSOR_AVAILABILITY
        self.classrooms = classrooms or DEFAULT_CLASSROOMS
        self.time_slots = time_slots or DEFAULT_TIME_SLOTS

        self.n_courses = len(self.courses)
        self.n_time_slots = len(self.time_slots)
        self.n_classrooms = len(self.classrooms)

    def evaluate(self, solution: list[dict[str, Any]]) -> float:
        """Evaluate a schedule solution.

        Computes penalty based on:
        - Professor unavailability
        - Classroom overcapacity
        - Professor time conflicts
        - Classroom double-booking
        - Student time conflicts

        Args:
            solution: List of assignment dicts with course, professor, time_slot, classroom.

        Returns:
            Total penalty (lower is better, 0 is optimal).
        """
        penalty = 0
        classroom_schedule: dict[str, list[int]] = {}
        professor_schedule: dict[str, list[int]] = {}
        student_schedule: dict[int, bool] = {}

        for assignment in solution:
            course_name = assignment["course"]
            time_slot = assignment["time_slot"]
            classroom_idx = assignment["classroom"]
            professor = assignment["professor"]

            classroom = self.classrooms[classroom_idx]
            course = next(c for c in self.courses if c["name"] == course_name)
            students = course["students"]

            # Check professor availability
            if not self.professor_availability[professor][time_slot]:
                penalty += 10
            else:
                # Check professor conflict
                if professor in professor_schedule and time_slot in professor_schedule[professor]:
                    penalty += 5
                else:
                    professor_schedule.setdefault(professor, []).append(time_slot)

            # Check classroom capacity
            if students > classroom["capacity"]:
                penalty += 5

            # Check classroom conflict
            classroom_name = classroom["name"]
            if classroom_name in classroom_schedule and time_slot in classroom_schedule[classroom_name]:
                penalty += 5
            else:
                classroom_schedule.setdefault(classroom_name, []).append(time_slot)

            # Check student conflict (simplified: all students same)
            if time_slot in student_schedule:
                penalty += 1
            else:
                student_schedule[time_slot] = True

        return penalty

    def check_conflicts(self, solution: list[dict[str, Any]]) -> list[str]:
        """Check and list all conflicts in a solution.

        Args:
            solution: Schedule solution to check.

        Returns:
            List of conflict description strings.
        """
        conflicts = []
        classroom_schedule: dict[str, list[int]] = {}
        professor_schedule: dict[str, list[int]] = {}
        student_schedule: dict[int, str] = {}

        for assignment in solution:
            course_name = assignment["course"]
            time_slot_idx = assignment["time_slot"]
            time_slot = self.time_slots[time_slot_idx]
            classroom = self.classrooms[assignment["classroom"]]
            professor = assignment["professor"]
            course = next(c for c in self.courses if c["name"] == course_name)
            students = course["students"]

            # Professor availability
            if not self.professor_availability[professor][time_slot_idx]:
                conflicts.append(
                    f"Professor {professor} unavailable at {time_slot} for {course_name}"
                )
            else:
                if professor in professor_schedule and time_slot_idx in professor_schedule[professor]:
                    conflicts.append(
                        f"Professor {professor} has multiple courses at {time_slot}"
                    )
                else:
                    professor_schedule.setdefault(professor, []).append(time_slot_idx)

            # Classroom capacity
            if students > classroom["capacity"]:
                conflicts.append(
                    f"Classroom {classroom['name']} overcapacity for {course_name}"
                )

            # Classroom conflict
            if classroom["name"] in classroom_schedule and time_slot_idx in classroom_schedule[classroom["name"]]:
                conflicts.append(
                    f"Classroom {classroom['name']} double-booked at {time_slot}"
                )
            else:
                classroom_schedule.setdefault(classroom["name"], []).append(time_slot_idx)

            # Student conflict
            if time_slot_idx in student_schedule:
                conflicts.append(
                    f"Students have multiple courses at {time_slot}"
                )
            else:
                student_schedule[time_slot_idx] = course_name

        return conflicts

    def format_schedule(self, solution: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Format solution for readable output.

        Args:
            solution: Raw schedule solution.

        Returns:
            List of formatted schedule entries.
        """
        formatted = []
        for assignment in solution:
            formatted.append({
                "course": assignment["course"],
                "professor": assignment["professor"],
                "time_slot": self.time_slots[assignment["time_slot"]],
                "classroom": self.classrooms[assignment["classroom"]]["name"],
            })
        return formatted

    def get_config(self) -> dict[str, Any]:
        """Get task configuration."""
        return {
            "name": self.name,
            "n_courses": self.n_courses,
            "n_time_slots": self.n_time_slots,
            "n_classrooms": self.n_classrooms,
        }
