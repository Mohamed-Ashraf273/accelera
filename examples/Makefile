ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
EXAMPLES := $(ROOT)/examples
TOOLS := $(ROOT)/tools
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

.PHONY: all run accpipe-demo parallel-accpipe code-parallelizer save-load clean-ram

all: run

run:
	@$(MAKE) -f "$(THIS_MAKEFILE)" --no-print-directory accpipe-demo
	@$(MAKE) -f "$(THIS_MAKEFILE)" --no-print-directory clean-ram
	@$(MAKE) -f "$(THIS_MAKEFILE)" --no-print-directory parallel-accpipe
	@$(MAKE) -f "$(THIS_MAKEFILE)" --no-print-directory clean-ram
	@$(MAKE) -f "$(THIS_MAKEFILE)" --no-print-directory code-parallelizer
	@$(MAKE) -f "$(THIS_MAKEFILE)" --no-print-directory clean-ram
	@$(MAKE) -f "$(THIS_MAKEFILE)" --no-print-directory save-load

accpipe-demo:
	@cd "$(ROOT)" && $(PYTHON) "$(EXAMPLES)/accpipe_demo.py"

parallel-accpipe:
	@cd "$(ROOT)" && $(PYTHON) "$(EXAMPLES)/parallel_accpipe.py"

code-parallelizer:
	@cd "$(ROOT)" && $(PYTHON) "$(EXAMPLES)/code_parallelizer_demo.py"

save-load:
	@cd "$(ROOT)" && $(PYTHON) "$(EXAMPLES)/save_load.py"

clean-ram:
	@cd "$(ROOT)" && $(PYTHON) "$(TOOLS)/clean_ram.py"
