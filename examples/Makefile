ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
EXAMPLES := $(ROOT)/examples
TOOLS := $(ROOT)/tools
BENCHMARK := $(ROOT)/accelera/src/benchmark
THIS_MAKEFILE := $(abspath $(lastword $(MAKEFILE_LIST)))
PYTHON ?= python

ifeq ($(OS),Windows_NT)
PATH_SEP := ;
EXEC_PATH_SEP := :
GRAPHVIZ_BIN ?= /c/Program Files/Graphviz/bin
else
PATH_SEP := :
EXEC_PATH_SEP := :
endif

export PYTHONPATH := $(ROOT)$(PATH_SEP)$(ROOT)/build/bindings$(PATH_SEP)$(PYTHONPATH)

ifeq ($(OS),Windows_NT)
export PATH := $(GRAPHVIZ_BIN)$(EXEC_PATH_SEP)$(PATH)
endif

.PHONY: all run accpipe-demo parallel-accpipe code-parallelizer save-load auto-preprocessing classification-automl regression-automl benchmark deployment-demo clean-ram

all: run

run:
	@$(MAKE) -f "$(THIS_MAKEFILE)" --no-print-directory accpipe-demo
	@$(MAKE) -f "$(THIS_MAKEFILE)" --no-print-directory clean-ram
	@$(MAKE) -f "$(THIS_MAKEFILE)" --no-print-directory parallel-accpipe
	@$(MAKE) -f "$(THIS_MAKEFILE)" --no-print-directory clean-ram
	@$(MAKE) -f "$(THIS_MAKEFILE)" --no-print-directory code-parallelizer
	@$(MAKE) -f "$(THIS_MAKEFILE)" --no-print-directory clean-ram
	@$(MAKE) -f "$(THIS_MAKEFILE)" --no-print-directory save-load
	@$(MAKE) -f "$(THIS_MAKEFILE)" --no-print-directory clean-ram
	@$(MAKE) -f "$(THIS_MAKEFILE)" --no-print-directory auto-preprocessing
	@$(MAKE) -f "$(THIS_MAKEFILE)" --no-print-directory clean-ram
	@$(MAKE) -f "$(THIS_MAKEFILE)" --no-print-directory classification-automl
	@$(MAKE) -f "$(THIS_MAKEFILE)" --no-print-directory clean-ram
	@$(MAKE) -f "$(THIS_MAKEFILE)" --no-print-directory regression-automl
	@$(MAKE) -f "$(THIS_MAKEFILE)" --no-print-directory clean-ram
	@$(MAKE) -f "$(THIS_MAKEFILE)" --no-print-directory deployment-demo
	@$(MAKE) -f "$(THIS_MAKEFILE)" --no-print-directory clean-ram
	@$(MAKE) -f "$(THIS_MAKEFILE)" --no-print-directory benchmark

accpipe-demo:
	@cd "$(ROOT)" && $(PYTHON) "$(EXAMPLES)/accpipe_demo.py"

parallel-accpipe:
	@cd "$(ROOT)" && $(PYTHON) "$(EXAMPLES)/parallel_accpipe.py"

code-parallelizer:
	@cd "$(ROOT)" && $(PYTHON) "$(EXAMPLES)/code_parallelizer_demo.py"

save-load:
	@cd "$(ROOT)" && $(PYTHON) "$(EXAMPLES)/save_load.py"

auto-preprocessing:
	@cd "$(ROOT)" && $(PYTHON) "$(EXAMPLES)/auto_preprocessing_demo.py"

classification-automl:
	@cd "$(ROOT)" && $(PYTHON) "$(EXAMPLES)/run_classification_task_automl.py"

regression-automl:
	@cd "$(ROOT)" && $(PYTHON) "$(EXAMPLES)/run_regression_task_automl.py"

deployment-demo:
	@cd "$(ROOT)" && $(PYTHON) "$(EXAMPLES)/deployment_demo.py"

benchmark:
	@cd "$(ROOT)" && $(PYTHON) "$(EXAMPLES)/run_benchmark_servers.py"

clean-ram:
	@$(PYTHON) "$(TOOLS)/clean_ram.py"
