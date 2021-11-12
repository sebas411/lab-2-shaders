
class Obj(object):
  def __init__(self, filename):
    with open(filename) as f:
      self.lines = f.read().splitlines()
    self.vertices = []
    self.faces = []
    self.read()
  
  def read(self):
    for line in self.lines:
      if line:
        prefix, value = line.split(' ', 1)

        if prefix == 'v':
          self.vertices.append(list(map(float, value.split(' '))))
          #self.vertices.append([float(x) for x in value.split(' ')])
        elif prefix == 'f':
          retList = []
          for face in value.split(' '):
            inList = []
            for f in face.split('/'):
              inList.append(int(f))
            retList.append(inList)
          self.faces.append(retList)
