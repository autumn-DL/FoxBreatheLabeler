from datetime import timedelta
from typing import Optional

from rich import filesize
from rich.console import JustifyMethod
from rich.highlighter import Highlighter
from rich.progress import Progress, ProgressColumn, Text, BarColumn, TaskProgressColumn, TimeElapsedColumn, TextColumn

from rich.style import StyleType
from rich.table import Column
from tqdm import tqdm


class SpeedColumn(TextColumn):

    def __init__(
            self,
            text_format: str = "[progress.percentage]{task.percentage:>3.0f}%",
            text_format_no_percentage: str = "",
            style: StyleType = "none",
            justify: JustifyMethod = "left",
            markup: bool = True,
            highlighter: Optional[Highlighter] = None,
            table_column: Optional[Column] = None,
            show_speed: bool = False,
    ) -> None:
        self.text_format_no_percentage = text_format_no_percentage
        self.show_speed = show_speed
        super().__init__(
            text_format=text_format,
            style=style,
            justify=justify,
            markup=markup,
            highlighter=highlighter,
            table_column=table_column,
        )

    @classmethod
    def render_speed(cls, speed: Optional[float]) -> Text:
        if speed is None:
            return Text("", style="progress.percentage")
        unit, suffix = filesize.pick_unit_and_suffix(
            int(speed),
            ["", "×10³", "×10⁶", "×10⁹", "×10¹²"],
            1000,
        )
        data_speed = speed / unit
        return Text(f"{data_speed:.1f}{suffix} it/s", style="progress.percentage")

    def render(self, task: "Task") -> Text:
        # if task.total is None and self.show_speed:
        return self.render_speed(task.finished_speed or task.speed)


class LCustomColumn(ProgressColumn):
    def __init__(self):
        super().__init__()
        self.text = {}

    def update_text(self, text_dict: dict, task_id: int):
        text = ''
        for i in text_dict:
            if isinstance(text_dict[i], float):
                texts = '{:.4f}'.format(text_dict[i])
            else:
                texts = text_dict[i]
            text = text + f'{str(i)}={str(texts)},'
        text = text[:-1]
        self.text[task_id] = text

    def render(self, task):

        temp_text = self.text.get(task.id)
        if temp_text is None:
            temp_text = ''
        return Text(f"{temp_text}", style="bold magenta")


class RCustomColumn(ProgressColumn):
    def __init__(self):
        super().__init__()
        self.text = {}

    def update_text(self, text, task_id: int):
        self.text[task_id] = text

    def render(self, task):
        temp_text = self.text.get(task.id)
        if temp_text is None:
            temp_text = ''
        return Text(f"{temp_text}", style="bold magenta")


class ShowItem(ProgressColumn):
    def __init__(self):
        super().__init__()
        self.text = ''

    def render(self, task):
        return Text(f"{task.completed}/{task.total}", style="bold magenta")


class TimeColumn(ProgressColumn):
    """Renders time elapsed."""

    def render(self, task: "Task") -> Text:
        """Show time elapsed."""
        elapsed = task.finished_time if task.finished else task.elapsed
        task_time = task.time_remaining
        if task_time is None:
            task_text = '-:--:--'
        else:
            task_text = timedelta(seconds=max(0, int(task_time)))
        if elapsed is None:
            elapsed_text = "-:--:--"
        else:
            elapsed_text = timedelta(seconds=max(0, int(elapsed)))

        return Text(f'{str(elapsed_text)}<{str(task_text)}', style="progress.elapsed")


class Adp_bar:
    def __init__(self, bar_type: str = 'tqdm'):

        self.bar_type = bar_type
        self.train_obj = None
        self.rich_obj = None
        self.rich_L_inf = None
        self.rich_R_inf = None
        self.train_task = None
        self.val_obj = None
        self.val_task = None
        self.rich_start = False
        if self.bar_type == 'rich':
            self.setup_rich()

    def setup_rich(self):
        self.rich_L_inf = LCustomColumn()
        self.rich_R_inf = RCustomColumn()
        self.rich_obj = Progress(self.rich_R_inf, BarColumn(), ShowItem(), TimeColumn(), SpeedColumn(),
                                 self.rich_L_inf)
        self.rich_obj.start()
        self.rich_start = True

    def setup_train(self, total: int):
        if self.bar_type == 'tqdm':
            self.train_obj = tqdm(total=total)
        elif self.bar_type == 'rich':
            if not self.rich_start:
                self.setup_rich()

            self.train_task = self.rich_obj.add_task("train", total=total)

    def update_train(self, num: int = 1):
        if self.bar_type == 'tqdm':
            self.train_obj.update(num)
        elif self.bar_type == 'rich':
            self.rich_obj.update(self.train_task, advance=num)

    def update_val(self, num: int = 1):
        if self.bar_type == 'tqdm':
            self.val_obj.update(num)
        elif self.bar_type == 'rich':
            self.rich_obj.update(self.val_task, advance=num)

    def rest_train(self):
        if self.bar_type == 'tqdm':
            self.train_obj.reset()
        elif self.bar_type == 'rich':
            self.rich_obj.reset(self.train_task)

    def setup_val(self, total: int):
        if self.bar_type == 'tqdm':
            self.val_obj = tqdm(total=total, leave=False)
        elif self.bar_type == 'rich':
            if not self.rich_start:
                self.setup_rich()
            self.val_task = self.rich_obj.add_task("val", total=total)

    def set_postfix_train(self, **key_arg):
        if self.bar_type == 'tqdm':
            self.train_obj.set_postfix(**key_arg)
        elif self.bar_type == 'rich':
            self.rich_L_inf.update_text(text_dict=key_arg, task_id=self.train_task)

    def set_postfix_val(self, **key_arg):
        if self.bar_type == 'tqdm':
            self.val_obj.set_postfix(**key_arg)
        elif self.bar_type == 'rich':
            self.rich_L_inf.update_text(text_dict=key_arg, task_id=self.val_task)

    def set_description_train(self, text):
        if self.bar_type == 'tqdm':
            self.train_obj.set_description(text)
        elif self.bar_type == 'rich':
            self.rich_R_inf.update_text(text=text, task_id=self.train_task)

    def set_description_val(self, text):
        if self.bar_type == 'tqdm':
            self.val_obj.set_description(text)
        elif self.bar_type == 'rich':
            self.rich_R_inf.update_text(text=text, task_id=self.val_task)

    def close_train(self):
        if self.bar_type == 'tqdm':
            self.train_obj.close()
        elif self.bar_type == 'rich':
            self.rich_obj.remove_task(self.train_task)

    def close_rich(self):
        if self.bar_type == 'tqdm':
            pass
        elif self.bar_type == 'rich':
            self.rich_obj.stop()
            self.rich_start = False

    def close_val(self):
        if self.bar_type == 'tqdm':
            self.val_obj.close()
        elif self.bar_type == 'rich':
            self.rich_obj.remove_task(self.val_task)
