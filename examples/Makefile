ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
EXAMPLES := $(ROOT)/examples
TOOLS := $(ROOT)/tools
THIS_MAKEFILE := $(abspath $(lastword $(MAKEFILE_LIST)))
PYTHON ?= python
ifeq ($(OS),Windows_NT)
PATH_SEP := ;
else
PATH_SEP := :
endif

export PYTHONPATH := $(ROOT)$(PATH_SEP)$(ROOT)/build/bindings$(PATH_SEP)$(PYTHONPATH)

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
	@$(PYTHON) "$(EXAMPLES)/accpipe_demo.py"

parallel-accpipe:
	@$(PYTHON) "$(EXAMPLES)/parallel_accpipe.py"

code-parallelizer:
	@$(PYTHON) "$(EXAMPLES)/code_parallelizer_demo.py"

save-load:
	@$(PYTHON) "$(EXAMPLES)/save_load.py"

clean-ram:
	@$(PYTHON) "$(TOOLS)/clean_ram.py"
