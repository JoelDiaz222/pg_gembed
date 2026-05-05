MODULE_big = pg_gembed
OBJS = src/pg_gembed.o \
       src/internal.o \
       src/embedding_worker.o

EXTENSION = pg_gembed
EXTVERSION = 1.0.0
DATA = sql/$(EXTENSION)--$(EXTVERSION).sql

PG_CONFIG = pg_config

PG_INCLUDEDIR = $(shell $(PG_CONFIG) --includedir-server)

VECTOR_INC_DIR ?= $(PG_INCLUDEDIR)/extension/vector

PG_CPPFLAGS = -I$(VECTOR_INC_DIR)

GEMBED_DIR = gembed
GEMBED_TARGET = $(GEMBED_DIR)/target/release
GEMBED_LIB = $(GEMBED_TARGET)/libgembed.a

UNAME_S := $(shell uname -s)

SHLIB_LINK = \
	-L$(GEMBED_TARGET) \
	-lgembed

# macOS specific flags
ifeq ($(UNAME_S),Darwin)
	SHLIB_LINK += -undefined dynamic_lookup
endif

# Windows/MSYS/MINGW specific flags
ifneq (,$(filter MINGW% MSYS%,$(UNAME_S)))
	SHLIB_LINK += -lntdll
endif

REGRESS = pg_gembed_test

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

$(shlib): $(GEMBED_LIB)

$(GEMBED_LIB):
	cd $(GEMBED_DIR) && cargo build --release

clean:
	rm -f $(OBJS) $(MODULE_big).so $(MODULE_big).dylib $(MODULE_big).dll
	cd $(GEMBED_DIR) && cargo clean
