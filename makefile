PROJECTS= \
	dijkstra \
	dijkstraOMP

.PHONY: all
all:
	$(foreach P,$(PROJECTS),$(MAKE) -C $(P$) &&) true


.PHONY: clean	
clean:
	$(foreach P,$(PROJECTS),$(MAKE) -C $(P) clean &&) true	
