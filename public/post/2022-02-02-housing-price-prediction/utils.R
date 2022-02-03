#' Create a Sorted Factor
sort_factor <- function (var, base.level = NULL, ...) {
  if (is.factor(var)) {
    warning("Input is already a factor - returning it unchanged.", 
            call. = FALSE)
    return(var)
  }
  if (!is.null(base.level)) {
    var <- droplevels(factor(var, levels = unique(c(base.level, 
                                                    sort(unique(var), na.last = TRUE))), ...))
    return(var)
  }
  else {
    var <- droplevels(factor(var, levels = sort(unique(var), 
                                                na.last = TRUE), ...))
    return(var)
  }
}

#' Open Python REPL in virtual environment
#'
#' \code{repl_python_virtualenv} Activates a Python virtual environment and
#' launches a Python REPL within that environment.
use_python_virtualenv <- function(envdir) {
  if (!check_win()) stop("Designed to be used on Windows only", call. = FALSE)
  venv_status <- py_venv_exists(envdir)
  if (!venv_status[[1]]) {
    stop(sprintf("Could not find %s", venv_status[[2]]), call. = FALSE)
  }
  Sys.setenv("RETICULATE_PYTHON" = venv_status[[2]])
}

#' Create Python virtual environment
#'
#' \code{virtualenv_create_win} creates a new Python virtual environment and
#' installs updated versions of pip, setuptools, and wheel in addition to any
#' other user-provided packages that are available on
#' \href{https://pypi.org/}{PyPI}.
virtualenv_create_win <- function(envdir, envname, packages = NULL) {
  if (!check_win()) stop("Designed to be used on Windows only", call. = FALSE)
  stopifnot(all(vapply(packages, is.character, logical(1))))
  b <- venv_create_body(envdir = envdir, envname = envname, packages = packages)
  writeLines(
    b,
    con = paste0(tempdir(), "\\makeNewVenv.ps1")
  )
  system2(
    "powershell",
    args = paste0(tempdir(), "\\makeNewVenv.ps1"),
    invisible = FALSE
  )
  invisible(NULL)
}

#' Translate filepaths to Windows style
#'
#' Filepaths in the Windows OS typically use `\` as a separator. This function
#' strips out all forward slashes and replaces them with back slashes.
win_dir <- function(path, check.exists = FALSE) {
  if (check.exists) {
    stopifnot(dir.exists(path))
  } else {
    stopifnot(!is.null(path) && is.character(path))
  }
  gsub("/", "\\\\", path.expand(path))
}

# Helpers -----------------------------------------------------------------

check_win <- function() {
  identical(.Platform$OS.type, "windows")
}

py_venv_exists <- function(path) {
  is_there <- file.exists(paste0(path, "/Scripts/python.exe"))
  a_file <- win_dir(paste0(path, "/Scripts/python.exe"))
  list(is_there, a_file)
}

venv_create_body <- function(envdir, envname, packages = NULL) {
  stopifnot(is.null(packages) || is.vector(packages), is.character(envname))
  envdir <- win_dir(envdir, check.exists = TRUE)
  packages <- append(c("pip", "setuptools", "wheel"), packages)
  paste0(
    c(
      "# Create virtual env",
      sprintf("cd %s", envdir),
      sprintf("python -m venv %s", envname),
      "# Set execution policy (if necessary) and activate environment",
      sprintf("try { .\\%s\\Scripts\\Activate.ps1 }", envname),
      "catch {",
      "  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser",
      sprintf("  .\\%s\\Scripts\\Activate.ps1", envname),
      "}",
      "# Install pip, setuptools, and wheel",
      paste("python -m pip install --upgrade --no-cache-dir", packages),
      "deactivate"
    ),
    collapse = "\n"
  )
}
