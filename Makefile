.PHONY : all
all:
	cd c2c_example && make $@
	cd r2c_c2r_example && make $@
	# cd c2r_r2c_example && make $@

.PHONY : clean
clean:
	cd c2c_example && make $@
	cd r2c_c2r_example && make $@
	# cd c2r_r2c_example && make $@