CC = gcc
CXX = g++
RM = rm -f
CPPFLAGS = -Wall -g 
LDFLAGS = -g 
LDLIBS = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_ml

SRCS = non_linear_svms.cpp additional_functions.cpp
OBJS = $(subst .cpp,.o,$(SRCS))

TARGET = program

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $(TARGET).exe $(OBJS) $(LDLIBS)

non_linear_svms.o: non_linear_svms.cpp definitions.h additional_functions.h

additional_functions.o: additional_functions.cpp definitions.h

clean:
	$(RM) $(OBJS)

deepclean: clean
	$(RM) $(TARGET).exe
