from collections import OrderedDict


class CalQueue(OrderedDict):
    __slots__ = ["insert"]
    
    def insert(self, key, value, after=None):
        # Validation
        for key, value in self.items():
            if value is not None:
                print(f"Analysis started ({key}), calibration queue is now immutable.")
        
        if after is None:
            self[key] = value
            
        else:
            new_dict = CalQueue()
            for _key, _value in self.items():
                
                if _key == after:
                    new_dict[_key] = _value
                    new_dict[key] = value
                
                else:
                   new_dict[_key] = _value

            return new_dict
