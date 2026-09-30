// SPDX-FileCopyrightText: 2023 Unity Technologies and the glTFast authors
// SPDX-License-Identifier: Apache-2.0

#if DRACO_UNITY

using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Draco;
using GLTFast.Logging;
using GLTFast.Schema;
using UnityEngine.Assertions;
using UnityEngine.Rendering;
using Mesh = UnityEngine.Mesh;

namespace GLTFast {

    class DracoMeshGenerator : MeshGeneratorBase {

        Bounds? m_Bounds;

        readonly bool m_NeedsNormals;
        readonly bool m_NeedsTangents;

        public DracoMeshGenerator(
            IReadOnlyList<MeshPrimitiveBase> primitives,
            SubMeshAssignment[] subMeshAssignments,
            string[] morphTargetNames,
            string meshName,
            GltfImportBase gltfImport
            )
            : base(meshName)
        {
            // subMeshAssignments is MeshGenerator's index-buffer-splitting mechanism (for meshes
            // exceeding the 16-bit index limit) — unrelated to multiple KHR_draco_mesh_compression
            // primitives merging into sub-meshes, which Decode() below does support.
            Assert.IsNull(
                subMeshAssignments,
                "Draco-compressed meshes requiring index-buffer-splitting sub-mesh assignment are not supported."
                );

            var morphTargets = primitives[0].targets;
            var hasMorphTargets = morphTargets != null && morphTargets.Length > 0;

            var vertexCount = 0;
            var vertexIntervals = hasMorphTargets
                ? new int[primitives.Count + 1]
                : null;

            for (var index = 0; index < primitives.Count; index++)
            {
                var primitive = primitives[index];
                Assert.IsTrue(primitive.IsDracoCompressed);

                var posAccessor = ((IGltfBuffers)gltfImport).GetAccessor(primitive.attributes.POSITION);

                if (hasMorphTargets)
                {
                    vertexIntervals[index] = vertexCount;
                }
                vertexCount += posAccessor.count;

                var bounds = posAccessor.TryGetBounds();

                if (bounds.HasValue)
                {
                    m_Bounds = bounds.Value;
                }
                else
                {
                    gltfImport.Logger?.Error(LogCode.MeshBoundsMissing, primitive.attributes.POSITION.ToString());
                }

                if (primitive.material < 0)
                {
                    m_NeedsNormals = true;
                }
                else
                {
                    var material = gltfImport.GetSourceMaterial(primitive.material);
                    m_NeedsNormals |= material.RequiresNormals;
                    m_NeedsTangents |= material.RequiresTangents;
                }
            }

            if (hasMorphTargets)
            {
                InitializeMorphTargets(
                    primitives,
                    morphTargetNames,
                    vertexIntervals,
                    vertexCount,
                    morphTargets,
                    gltfImport
                    );
            }

            m_CreationTask = Decode(primitives, gltfImport);
        }

        void InitializeMorphTargets(
            IReadOnlyList<MeshPrimitiveBase> primitives,
            string[] morphTargetNames,
            int[] vertexIntervals,
            int vertexCount,
            MorphTarget[] morphTargets,
            GltfImportBase gltfImport
            )
        {
            vertexIntervals[vertexIntervals.Length-1] = vertexCount;
            m_MorphTargetsGenerator = new MorphTargetsGenerator(
                vertexCount,
                1,
                morphTargets.Length,
                morphTargetNames,
                morphTargets[0].NORMAL >= 0,
                morphTargets[0].TANGENT >= 0,
                gltfImport
            );
            for (var subMesh = 0; subMesh < primitives.Count; subMesh++)
            {
                var primitive = primitives[subMesh];
                for (var morphTargetIndex = 0; morphTargetIndex < primitive.targets.Length; morphTargetIndex++)
                {
                    var target = primitive.targets[morphTargetIndex];
                    m_MorphTargetsGenerator.AddMorphTarget( 0, morphTargetIndex, target);
                }
            }
        }

        async Task<Mesh> Decode(IReadOnlyList<MeshPrimitiveBase> primitives, IGltfBuffers buffers)
        {
            // One buffer + attribute map per primitive, decoded together into a single mesh
            // with one sub-mesh per primitive. Decoding primitives one at a time here and
            // keeping only the last result silently dropped every earlier primitive's geometry
            // while sub-mesh/vertex-count bookkeeping (computed from ALL primitives, above)
            // stayed as if it hadn't been — sub-meshes ended up indexing into a vertex buffer
            // that only actually held the last primitive's data.
            var encodedBuffers = new List<NativeArray<byte>.ReadOnly>(primitives.Count);
            var attributeIdMaps = new List<Dictionary<VertexAttribute, int>>(primitives.Count);
            foreach (var primitive in primitives)
            {
                var dracoExt = primitive.Extensions.KHR_draco_mesh_compression;
                var buffer = buffers.GetBufferView(dracoExt.bufferView, out _);
                encodedBuffers.Add(AsNativeArrayReadOnly(buffer));
                attributeIdMaps.Add(GenerateAttributeIdMap(dracoExt.attributes));
            }

            var mesh = await StartDecode(encodedBuffers, attributeIdMaps);

            if (mesh is null) {
                return null;
            }

            if (m_Bounds.HasValue) {
                mesh.bounds = m_Bounds.Value;

                // Setting the sub-meshes' bounds to the overall bounds
                // Calculating the actual sub-mesh bounds (by iterating the verts referenced
                // by the sub-mesh indices) would be slow. Also, hardly any glTFs re-use
                // the same vertex buffer across primitives of a node (which is the
                // only way a mesh can have sub-meshes)
                for (var i = 0; i < mesh.subMeshCount; i++) {
                    var subMeshDescriptor = mesh.GetSubMesh(i);
                    subMeshDescriptor.bounds = m_Bounds.Value;
                    mesh.SetSubMesh(
                        i,
                        subMeshDescriptor,
                        MeshUpdateFlags.DontValidateIndices
                        | MeshUpdateFlags.DontResetBoneBounds
                        | MeshUpdateFlags.DontNotifyMeshUsers
                        | MeshUpdateFlags.DontRecalculateBounds
                        );
                }
            } else {
                mesh.RecalculateBounds();
            }

            if (m_MorphTargetsGenerator != null) {
                await m_MorphTargetsGenerator.ApplyOnMeshAndDispose(mesh);
            }

            mesh.name = m_MeshName;

#if GLTFAST_KEEP_MESH_DATA
            mesh.UploadMeshData(false);
#endif

            return mesh;
        }

        async Task<Mesh> StartDecode(
            IReadOnlyList<NativeArray<byte>.ReadOnly> data,
            IReadOnlyList<Dictionary<VertexAttribute, int>> attributeIdMaps
            )
        {
            var flags = DecodeSettings.ConvertSpace;
            if (m_NeedsTangents)
            {
                flags |= DecodeSettings.RequireNormalsAndTangents;
            }
            else if (m_NeedsNormals)
            {
                flags |= DecodeSettings.RequireNormals;
            }
            // Quantized attributes (e.g. UVs) can decode to a non-Float32 layout that
            // downstream mesh-building code can't read, causing "Unsupported conversion
            // of vertex data" errors. Force the compatible layout unconditionally, not
            // just when morph targets are present.
            flags |= DecodeSettings.ForceUnityVertexLayout;

            return await DracoDecoder.DecodeMesh(data, flags, attributeIdMaps);
        }

        // Draco-for-Unity has the equivalent conversion (NativeSliceExtensions.AsNativeArray), but
        // that type is internal to its own assembly and not visible here. Zero-copy reinterpretation
        // of the slice's existing memory as a NativeArray, same as that internal implementation.
        static unsafe NativeArray<byte>.ReadOnly AsNativeArrayReadOnly(NativeSlice<byte> slice)
        {
            var array = NativeArrayUnsafeUtility.ConvertExistingDataToNativeArray<byte>(
                slice.GetUnsafeReadOnlyPtr(),
                slice.Length,
                Allocator.None
                );
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            NativeArrayUnsafeUtility.SetAtomicSafetyHandle(ref array, AtomicSafetyHandle.Create());
#endif
            return array.AsReadOnly();
        }

        static Dictionary<VertexAttribute, int> GenerateAttributeIdMap(Attributes attributes)
        {
            var result = new Dictionary<VertexAttribute, int>();
            if (attributes.POSITION >= 0)
                result[VertexAttribute.Position] = attributes.POSITION;
            if (attributes.NORMAL >= 0)
                result[VertexAttribute.Normal] = attributes.NORMAL;
            if (attributes.TANGENT >= 0)
                result[VertexAttribute.Tangent] = attributes.TANGENT;
            if (attributes.COLOR_0 >= 0)
                result[VertexAttribute.Color] = attributes.COLOR_0;
            if (attributes.TEXCOORD_0 >= 0)
                result[VertexAttribute.TexCoord0] = attributes.TEXCOORD_0;
            if (attributes.TEXCOORD_1 >= 0)
                result[VertexAttribute.TexCoord1] = attributes.TEXCOORD_1;
            if (attributes.TEXCOORD_2 >= 0)
                result[VertexAttribute.TexCoord2] = attributes.TEXCOORD_2;
            if (attributes.TEXCOORD_3 >= 0)
                result[VertexAttribute.TexCoord3] = attributes.TEXCOORD_3;
            if (attributes.TEXCOORD_4 >= 0)
                result[VertexAttribute.TexCoord4] = attributes.TEXCOORD_4;
            if (attributes.TEXCOORD_5 >= 0)
                result[VertexAttribute.TexCoord5] = attributes.TEXCOORD_5;
            if (attributes.TEXCOORD_6 >= 0)
                result[VertexAttribute.TexCoord6] = attributes.TEXCOORD_6;
            if (attributes.TEXCOORD_7 >= 0)
                result[VertexAttribute.TexCoord7] = attributes.TEXCOORD_7;
            if (attributes.WEIGHTS_0 >= 0)
                result[VertexAttribute.BlendWeight] = attributes.WEIGHTS_0;
            if (attributes.JOINTS_0 >= 0)
                result[VertexAttribute.BlendIndices] = attributes.JOINTS_0;
            return result;
        }
    }
}
#endif // DRACO_UNITY
