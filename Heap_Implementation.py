#Heap Implementation

import heapq

hospital = []

heapq.heappush(hospital, (2, "Patient A : Dog Bite"))
heapq.heappush(hospital, (5, "Patient B : Cold"))
heapq.heappush(hospital, (1, "Patient C : Heart Attack"))
heapq.heappush(hospital, (4, "Patient D : Fracture"))   
heapq.heappush(hospital, (3, "Patient E : High Fever"))

print("Hospital Queue (by priority):", hospital)


while hospital:
    priority, patient = heapq.heappop(hospital)
    print("Attending to:", patient, "with priority:", priority)