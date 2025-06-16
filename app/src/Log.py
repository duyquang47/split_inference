""" Module xử lý logging với màu sắc """

class Colors:
    COLORS = {
        "header": '\033[95m',
        "blue": '\033[94m',
        "green": '\033[92m',
        "yellow": '\033[93m',
        "red": '\033[91m',
        "end": '\033[0m'
    }

def print_with_color(text: str, color: str) -> None:
    """
    In text với màu sắc.
    Args:
        text: Text cần in
        color: Màu sắc (header, blue, green, yellow, red)
    """
    color_code = Colors.COLORS.get(color.lower(), Colors.COLORS["end"])
    print(f"{color_code}{text}{Colors.COLORS['end']}")

def log_success(text: str) -> None:
    """ Success log with green color """
    print_with_color(f"[SUCCESS] {text}", "green")

def log_info(text: str) -> None:
    """ Info log with blue color """
    print_with_color(f"[INFO] {text}", "blue")

def log_warning(text: str) -> None:
    """ Warning log with yellow color """
    print_with_color(f"[WARN] {text}", "yellow")

def log_error(text: str) -> None:
    """ Error log with red color """
    print_with_color(f"[ERROR] {text}", "red")

def log_debug(text: str, debug_mode: bool = False) -> None:
    """ Debug log with green color if debug_mode=True """
    if debug_mode:
        print_with_color(f"[DEBUG] {text}", "green")
